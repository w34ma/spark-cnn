#include "matrix_c.h"

void im2col_c(double* X0, double* XC, int N, int W, int H, int D, int F, int S, int P) {
    /* convert matrix to stretched columns
     X0: input matrix, zero padded [N x (W + 2P) x (H + 2P) x D]
     W:  original width
     H:  original height
     D:  original depth
     F:  filter size (assume same width and height)
     S:  filter stride
     P:  zero paddings
     output XC: [(N x W_ x H_) x (F x F x D)] */

    // compute dimensions
    int W_ = (W - F + 2 * P) / S + 1;
    int H_ = (H - F + 2 * P) / S + 1;
    W = W + 2 * P;
    H = H + 2 * P;

    // loop through all small filter matrices and update stretched column matrix accordingly
    int n, w, h, fw, fh, d;
    for (n = 0; n < N; n++) {
        for (w = 0; w < W_; w++) {
            for (h = 0; h < H_; h++) {
                for (fw = 0; fw < F; fw++) {
                    for (fh = 0; fh < F; fh++) {
                        for (d = 0; d < D; d++) {
                            int fromPosition = n * (W * H * D) + (w * S + fw) * (H * D) + (h * S + fh) * D + d;
                            int toPosition = (n * W_ * H_ + w * H_ + h) * F * F * D + fw * F * D + fh * D + d;
                            XC[toPosition] = X0[fromPosition];
                        }
                    }
                }
            }
        }
    }
    return;
}

void col2im_c(double* dXC, double* dX0,
int N, int W, int H, int D, int F, int S, int P) {

    return;
}
