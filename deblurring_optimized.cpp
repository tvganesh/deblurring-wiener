/*
============================================================================
Name    : deblurring.cpp
Author  : Tinniam V Ganesh & Egli Simon
Description : Blind deconvolution via iterative kernel search.
              Tests circular, Gaussian, and motion blur kernel families.
              For each family the parameters are optimised by minimising a
              composite loss:
                  L(K) = MSE(K * F_hat, G)  -  lambda * Var(Laplacian(F_hat))
              where F_hat = Wiener(G, K).
              The best kernel across all three families is selected and the
              final deblurred image is displayed.
============================================================================
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <limits>
#include <functional>
#include <opencv2/opencv.hpp>

// ---- Tunable constants -------------------------------------------------
static const double KAPPA  = 100.0;  // Wiener regularisation — tuned by EPOCH (was 0.01)
static const double LAMBDA = 0.005;  // weight of sharpness reward in loss

// ========================================================================
// Kernel builders
// ========================================================================

// Uniform circular (disk) kernel of given pixel radius
cv::Mat buildCircularKernel(int radius)
{
    int sz = 2 * radius + 1;
    cv::Mat K = cv::Mat::zeros(sz, sz, CV_32F);
    int c = radius, count = 0;
    for (int y = 0; y < sz; y++)
        for (int x = 0; x < sz; x++)
            if (std::sqrt((double)(x-c)*(x-c) + (double)(y-c)*(y-c)) <= radius) {
                K.at<float>(y, x) = 1.0f;
                count++;
            }
    if (count > 0) K /= (float)count;
    return K;
}

// Normalised Gaussian kernel
cv::Mat buildGaussianKernel(double sigma)
{
    int half = (int)std::ceil(3.0 * sigma);
    int sz   = 2 * half + 1;
    cv::Mat K(sz, sz, CV_32F);
    double sum = 0.0;
    for (int y = 0; y < sz; y++)
        for (int x = 0; x < sz; x++) {
            double v = std::exp(-((x-half)*(x-half) + (y-half)*(y-half))
                                 / (2.0 * sigma * sigma));
            K.at<float>(y, x) = (float)v;
            sum += v;
        }
    K /= (float)sum;
    return K;
}

// Linear motion-blur kernel: a line of given length at given angle (degrees)
cv::Mat buildMotionBlurKernel(int length, double angleDeg)
{
    int sz    = (length % 2 == 0) ? length + 1 : length;
    cv::Mat K = cv::Mat::zeros(sz, sz, CV_32F);
    double angle = angleDeg * CV_PI / 180.0;
    double cx = (sz - 1) / 2.0, cy = (sz - 1) / 2.0;

    for (int i = 0; i < length; i++) {
        double t = (length > 1) ? ((double)i / (length - 1) - 0.5) : 0.0;
        int x = (int)std::round(cx + t * (length - 1) * std::cos(angle));
        int y = (int)std::round(cy + t * (length - 1) * std::sin(angle));
        if (x >= 0 && x < sz && y >= 0 && y < sz)
            K.at<float>(y, x) = 1.0f;
    }
    double s = cv::sum(K)[0];
    if (s > 0) K /= (float)s;
    return K;
}

// ========================================================================
// Wiener deconvolution (frequency domain)
// Returns recovered image normalised to [0,1], type CV_64F
// ========================================================================

cv::Mat applyWiener(const cv::Mat& G, const cv::Mat& K, double kappa)
{
    int dft_M = cv::getOptimalDFTSize(G.rows + K.rows - 1);
    int dft_N = cv::getOptimalDFTSize(G.cols + K.cols - 1);

    // Build a zero-padded complex DFT matrix from a real source image
    auto makeDFT = [&](const cv::Mat& src) -> cv::Mat {
        cv::Mat sf; src.convertTo(sf, CV_64F);
        cv::Mat si = cv::Mat::zeros(sf.size(), CV_64F);
        cv::Mat sc; cv::merge(std::vector<cv::Mat>{sf, si}, sc);
        cv::Mat out = cv::Mat::zeros(dft_M, dft_N, CV_64FC2);
        sc.copyTo(out(cv::Rect(0, 0, src.cols, src.rows)));
        cv::dft(out, out, cv::DFT_COMPLEX_OUTPUT);
        return out;
    };

    cv::Mat dft_G = makeDFT(G);
    cv::Mat dft_K = makeDFT(K);

    // Numerator:  G_hat * H_conj
    cv::Mat dft_F;
    cv::mulSpectrums(dft_G, dft_K, dft_F, 0, /*conjB=*/true);

    // Denominator: |H|^2 + kappa
    std::vector<cv::Mat> pK(2), pF(2);
    cv::split(dft_K, pK);
    cv::split(dft_F, pF);

    cv::Mat reK2, imK2, denom;
    cv::pow(pK[0], 2.0, reK2);
    cv::pow(pK[1], 2.0, imK2);
    cv::add(reK2, imK2, denom);
    cv::add(denom, cv::Scalar::all(kappa), denom);

    cv::divide(pF[0], denom, pF[0]);
    cv::divide(pF[1], denom, pF[1]);

    cv::Mat dft_filtered;
    cv::merge(pF, dft_filtered);

    // Inverse DFT → extract real part → crop to original size
    cv::Mat result;
    cv::dft(dft_filtered, result, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);
    std::vector<cv::Mat> rp(2);
    cv::split(result, rp);
    cv::Mat F_hat = rp[0](cv::Rect(0, 0, G.cols, G.rows)).clone();

    // Normalise to [0, 1]
    double mn, mx;
    cv::minMaxLoc(F_hat, &mn, &mx);
    if (mx > mn) F_hat = (F_hat - mn) / (mx - mn);
    return F_hat;
}

// ========================================================================
// Loss function
//   L(K) = MSE(K * F_hat, G)  -  lambda * Var(Laplacian(F_hat))
//
// First term:  reconstruction consistency — re-blurring the recovered image
//              with K should reproduce the original blurry input G.
// Second term: sharpness reward — F_hat with higher Laplacian variance is
//              sharper; subtracting it steers the search toward kernels that
//              produce crisper results.
// ========================================================================

double computeLoss(const cv::Mat& G, const cv::Mat& K, double kappa, double lambda)
{
    cv::Mat F_hat = applyWiener(G, K, kappa);  // CV_64F, range [0,1]

    // Re-blur recovered image with K in CV_32F (filter2D requires matching types)
    cv::Mat F32, Kf, G_hat;
    F_hat.convertTo(F32, CV_32F);
    K.convertTo(Kf, CV_32F);
    cv::filter2D(F32, G_hat, CV_32F, Kf, cv::Point(-1,-1), 0, cv::BORDER_REFLECT);

    // Normalise blurry input to [0,1] in CV_32F for a fair MSE comparison
    cv::Mat G_norm; G.convertTo(G_norm, CV_32F);
    double mn, mx;
    cv::minMaxLoc(G_norm, &mn, &mx);
    if (mx > mn) G_norm = (G_norm - mn) / (mx - mn);

    // Reconstruction MSE
    cv::Mat diff;
    cv::subtract(G_hat, G_norm, diff);
    cv::pow(diff, 2.0, diff);
    double recon = cv::mean(diff)[0];

    // Sharpness: variance of the Laplacian of F_hat
    cv::Mat lap;
    cv::Laplacian(F32, lap, CV_32F);
    cv::Scalar mean_l, std_l;
    cv::meanStdDev(lap, mean_l, std_l);
    double sharpness = std_l[0] * std_l[0];

    return recon - lambda * sharpness;
}

// ========================================================================
// Per-family optimisers
// ========================================================================

struct KernelResult {
    std::string name;
    cv::Mat     kernel;
    double      loss;
    std::string desc;
};

// Full grid search over integer radii 1..20
KernelResult optimiseCircular(const cv::Mat& G)
{
    printf("\n--- Circular kernel search ---\n");
    double best_loss = std::numeric_limits<double>::max();
    int best_r = 1;

    for (int r = 1; r <= 20; r++) {
        cv::Mat K   = buildCircularKernel(r);
        double loss = computeLoss(G, K, KAPPA, LAMBDA);
        printf("  radius=%-3d  loss=%+.6f\n", r, loss);
        if (loss < best_loss) { best_loss = loss; best_r = r; }
    }

    cv::Mat K = buildCircularKernel(best_r);
    char desc[64];
    snprintf(desc, sizeof(desc), "Circular r=%d", best_r);
    printf("  => Best: %s  loss=%+.6f\n", desc, best_loss);
    return { "circular", K, best_loss, desc };
}

// Grid search over sigma in [0.5, 15.0] in steps of 0.5
KernelResult optimiseGaussian(const cv::Mat& G)
{
    printf("\n--- Gaussian kernel search ---\n");
    double best_loss  = std::numeric_limits<double>::max();
    double best_sigma = 0.5;

    for (double sigma = 0.5; sigma <= 15.0; sigma += 0.5) {
        cv::Mat K   = buildGaussianKernel(sigma);
        double loss = computeLoss(G, K, KAPPA, LAMBDA);
        printf("  sigma=%-5.2f  loss=%+.6f\n", sigma, loss);
        if (loss < best_loss) { best_loss = loss; best_sigma = sigma; }
    }

    cv::Mat K = buildGaussianKernel(best_sigma);
    char desc[64];
    snprintf(desc, sizeof(desc), "Gaussian sigma=%.2f", best_sigma);
    printf("  => Best: %s  loss=%+.6f\n", desc, best_loss);
    return { "gaussian", K, best_loss, desc };
}

// Coarse grid search over (length, angle), then fine search around best
KernelResult optimiseMotionBlur(const cv::Mat& G)
{
    printf("\n--- Motion blur kernel search ---\n");
    double best_loss  = std::numeric_limits<double>::max();
    int    best_len   = 5;
    double best_angle = 0.0;

    // Coarse: length 3..25 step 4, angle 0..150 step 30
    printf("  [Coarse pass]\n");
    for (int len = 3; len <= 25; len += 4)
        for (double ang = 0.0; ang < 180.0; ang += 30.0) {
            cv::Mat K   = buildMotionBlurKernel(len, ang);
            double loss = computeLoss(G, K, KAPPA, LAMBDA);
            printf("  len=%-3d  angle=%-6.1f  loss=%+.6f\n", len, ang, loss);
            if (loss < best_loss) { best_loss = loss; best_len = len; best_angle = ang; }
        }

    // Fine: zoom in around coarse winner
    printf("  [Fine pass around len=%d, angle=%.1f]\n", best_len, best_angle);
    for (int len = std::max(2, best_len - 3); len <= best_len + 3; len++)
        for (double ang  = std::max(0.0,   best_angle - 25.0);
                   ang  <= std::min(175.0, best_angle + 25.0);
                   ang  += 5.0) {
            cv::Mat K   = buildMotionBlurKernel(len, ang);
            double loss = computeLoss(G, K, KAPPA, LAMBDA);
            printf("  len=%-3d  angle=%-6.1f  loss=%+.6f\n", len, ang, loss);
            if (loss < best_loss) { best_loss = loss; best_len = len; best_angle = ang; }
        }

    cv::Mat K = buildMotionBlurKernel(best_len, best_angle);
    char desc[64];
    snprintf(desc, sizeof(desc), "Motion len=%d angle=%.1f", best_len, best_angle);
    printf("  => Best: %s  loss=%+.6f\n", desc, best_loss);
    return { "motion", K, best_loss, desc };
}

// ========================================================================
// Main
// ========================================================================

// Apply the winning kernel to the Y channel of a colour image and return
// the fully deblurred colour image in BGR.
cv::Mat deblurColor(const cv::Mat& bgrImage, const cv::Mat& K, double kappa)
{
    // Work in YCrCb so chrominance (colour) is untouched
    cv::Mat ycrcb;
    cv::cvtColor(bgrImage, ycrcb, cv::COLOR_BGR2YCrCb);

    std::vector<cv::Mat> ch(3);
    cv::split(ycrcb, ch);   // ch[0]=Y, ch[1]=Cr, ch[2]=Cb

    // Deblur luminance channel and scale back to [0, 255]
    cv::Mat Y_deblurred = applyWiener(ch[0], K, kappa);  // [0,1] CV_64F
    Y_deblurred *= 255.0;
    Y_deblurred.convertTo(ch[0], CV_8U);

    cv::Mat result_ycrcb;
    cv::merge(ch, result_ycrcb);

    cv::Mat result_bgr;
    cv::cvtColor(result_ycrcb, result_bgr, cv::COLOR_YCrCb2BGR);
    return result_bgr;
}

int main(int argc, char** argv)
{
    // Grayscale copy — used for all kernel search / loss computation
    cv::Mat im = cv::imread("kutty-1.jpg", cv::IMREAD_GRAYSCALE);
    if (im.empty()) {
        printf("Error: could not load 'kutty-1.jpg'\n");
        return -1;
    }

    // Colour copy — used only for the final colour output
    cv::Mat imColor = cv::imread("kutty-1.jpg", cv::IMREAD_COLOR);

    cv::namedWindow("Input image (color)", cv::WINDOW_NORMAL);
    cv::imshow("Input image (color)", imColor);

    // ---- Run optimisation for each kernel family -----------------------
    std::vector<KernelResult> results;
    results.push_back(optimiseCircular(im));
    results.push_back(optimiseGaussian(im));
    results.push_back(optimiseMotionBlur(im));

    // ---- Find the global winner ----------------------------------------
    int best_idx = 0;
    for (int i = 1; i < (int)results.size(); i++)
        if (results[i].loss < results[best_idx].loss)
            best_idx = i;

    // ---- Print summary -------------------------------------------------
    printf("\n========================================\n");
    printf("RESULTS SUMMARY:\n");
    for (int i = 0; i < (int)results.size(); i++)
        printf("  %-35s  loss=%+.6f%s\n",
               results[i].desc.c_str(), results[i].loss,
               (i == best_idx) ? "  <-- WINNER" : "");
    printf("========================================\n");

    // ---- Display grayscale deblurred result for every kernel type ------
    for (auto& r : results) {
        cv::Mat F = applyWiener(im, r.kernel, KAPPA);
        cv::namedWindow("Gray: " + r.desc, cv::WINDOW_NORMAL);
        cv::imshow("Gray: " + r.desc, F);
    }

    // ---- Display colour deblurred result for every kernel type ---------
    for (auto& r : results) {
        cv::Mat Fc = deblurColor(imColor, r.kernel, KAPPA);
        cv::namedWindow("Color: " + r.desc, cv::WINDOW_NORMAL);
        cv::imshow("Color: " + r.desc, Fc);
    }

    // ---- Highlight the winner in colour --------------------------------
    cv::Mat best_color = deblurColor(imColor, results[best_idx].kernel, KAPPA);
    std::string title  = "BEST (color): " + results[best_idx].desc;
    cv::namedWindow(title, cv::WINDOW_NORMAL);
    cv::imshow(title, best_color);

    cv::waitKey(0);
    return 0;
}
