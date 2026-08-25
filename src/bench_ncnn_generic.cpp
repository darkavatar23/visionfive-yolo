// Generic NCNN bench: pure forward on a dummy input, any model.
// Usage: bench_ncnn_generic <prefix> <imgsz> <n_iter> <threads> [in_name] [out_name]
// Auto-detects input/output from the graph. Prints median ms/frame.
#include "net.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <chrono>
using namespace std::chrono;

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s prefix imgsz n threads [in] [out]\n", argv[0]); return 1; }
    std::string prefix = argv[1];
    int imgsz = atoi(argv[2]), N = atoi(argv[3]), th = atoi(argv[4]);
    ncnn::Net net;
    net.opt.num_threads = th;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_packing_layout = true;
    net.opt.use_sgemm_convolution = true;
    net.opt.use_winograd_convolution = true;
    if (net.load_param((prefix + ".param").c_str())) { fprintf(stderr, "param load failed\n"); return 2; }
    if (net.load_model((prefix + ".bin").c_str()))   { fprintf(stderr, "bin load failed\n"); return 2; }
    std::string in = argc > 5 ? argv[5] : (net.input_names().size() ? net.input_names()[0] : "in0");
    std::string out = argc > 6 ? argv[6] : (net.output_names().size() ? net.output_names().back() : "out0");

    ncnn::Mat inm(imgsz, imgsz, 3);
    inm.fill(0.5f);
    // warmup
    for (int i = 0; i < 3; i++) {
        ncnn::Extractor ex = net.create_extractor();
        ex.input(in.c_str(), inm); ncnn::Mat o; ex.extract(out.c_str(), o);
    }
    std::vector<double> ts;
    for (int i = 0; i < N; i++) {
        auto t0 = high_resolution_clock::now();
        ncnn::Extractor ex = net.create_extractor();
        ex.input(in.c_str(), inm);
        ncnn::Mat o; ex.extract(out.c_str(), o);
        auto t1 = high_resolution_clock::now();
        ts.push_back(duration_cast<microseconds>(t1 - t0).count() / 1000.0);
    }
    std::sort(ts.begin(), ts.end());
    double med = ts[ts.size() / 2];
    printf("%-22s imgsz=%d  %.0f ms  %.2f FPS  (in=%s out=%s)\n",
           prefix.c_str(), imgsz, med, 1000.0 / med, in.c_str(), out.c_str());
    return 0;
}
