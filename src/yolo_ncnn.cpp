// YOLOv8/YOLO11/YOLO26 detector on NCNN, STATIC binary (no glibc/python deps).
// Protocol (long-running, model loaded once):
//   stdin:  "<raw_path> <W> <H>\n"   raw BGR frame (W*H*3 bytes) at raw_path
//   stdout: "<n>\n" then n lines "<cls> <conf> <x1> <y1> <x2> <y2>\n" (original-frame coords)
// Model: <prefix>.param / <prefix>.bin  (ultralytics ncnn export, classic-NMS v8/11 head)
// Usage: yolo_ncnn <model_prefix> <imgsz> <conf> <nms> <threads> [fp16=0]
#include "net.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

struct Det { int cls; float conf; float x1, y1, x2, y2; };

static float iou(const Det& a, const Det& b) {
    float xx1 = std::max(a.x1, b.x1), yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2), yy2 = std::min(a.y2, b.y2);
    float w = std::max(0.f, xx2 - xx1), h = std::max(0.f, yy2 - yy1);
    float inter = w * h;
    float ua = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
    return ua > 0 ? inter / ua : 0.f;
}

int main(int argc, char** argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s prefix imgsz conf nms threads [fp16=0]\n", argv[0]); return 1; }
    std::string prefix = argv[1];
    int imgsz = atoi(argv[2]);
    float conf_th = atof(argv[3]);
    float nms_th = atof(argv[4]);
    int threads = atoi(argv[5]);
    bool fp16 = (argc > 6) ? atoi(argv[6]) != 0 : false;   // fp16 storage/packed (no hw arithmetic)

    ncnn::Net net;
    net.opt.num_threads = threads;
    net.opt.use_fp16_packed = fp16;
    net.opt.use_fp16_storage = fp16;
    net.opt.use_fp16_arithmetic = false;   // U74 has no RVV: no fp16 arithmetic
    net.opt.use_packing_layout = true;
    net.opt.use_sgemm_convolution = true;
    net.opt.use_winograd_convolution = true;
    if (net.load_param((prefix + ".param").c_str())) { fprintf(stderr, "param load failed\n"); return 2; }
    if (net.load_model((prefix + ".bin").c_str()))   { fprintf(stderr, "bin load failed\n"); return 2; }

    // resolve input/output blob names from the graph
    std::string in_name = net.input_names().size() ? net.input_names()[0] : "in0";
    std::string out_name = net.output_names().size() ? net.output_names().back() : "out0";

    fprintf(stderr, "READY %s in=%s out=%s imgsz=%d thr=%d\n",
            prefix.c_str(), in_name.c_str(), out_name.c_str(), imgsz, threads);
    fflush(stderr);

    char line[1024];
    std::vector<unsigned char> buf;
    while (fgets(line, sizeof(line), stdin)) {
        char path[900]; int W = 0, H = 0;
        if (sscanf(line, "%899s %d %d", path, &W, &H) != 3 || W <= 0 || H <= 0) {
            printf("0\n"); fflush(stdout); continue;
        }
        FILE* f = fopen(path, "rb");
        if (!f) { printf("0\n"); fflush(stdout); continue; }
        size_t need = (size_t)W * H * 3;
        buf.resize(need);
        size_t got = fread(buf.data(), 1, need, f);
        fclose(f);
        if (got != need) { printf("0\n"); fflush(stdout); continue; }

        // letterbox -> imgsz x imgsz
        float r = std::min((float)imgsz / W, (float)imgsz / H);
        int nw = (int)std::round(W * r), nh = (int)std::round(H * r);
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(
            buf.data(), ncnn::Mat::PIXEL_BGR2RGB, W, H, nw, nh);
        int dw = imgsz - nw, dh = imgsz - nh;
        ncnn::Mat inpad;
        ncnn::copy_make_border(in, inpad, 0, dh, 0, dw,
                               ncnn::BORDER_CONSTANT, 114.f);
        const float norm[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
        inpad.substract_mean_normalize(0, norm);

        ncnn::Extractor ex = net.create_extractor();
        ex.input(in_name.c_str(), inpad);
        ncnn::Mat out;
        ex.extract(out_name.c_str(), out);

        // out layout: [num_features, num_anchors], features = 4 bbox + 80 classes (v8/11/26)
        // ncnn Mat: usually w = num_anchors, h = num_features (84)
        int na, nf;
        // handle both layouts
        if (out.h == 84 || out.h == 85) { nf = out.h; na = out.w; }
        else { nf = out.w; na = out.h; }
        int ncls = nf - 4;

        std::vector<Det> cand;
        for (int i = 0; i < na; i++) {
            float cx, cy, bw, bh;
            const float* col;
            if (out.h == 84 || out.h == 85) {
                // features on rows: out.row(f)[i]
                cx = out.row(0)[i]; cy = out.row(1)[i];
                bw = out.row(2)[i]; bh = out.row(3)[i];
                float best = 0; int bc = -1;
                for (int c = 0; c < ncls; c++) {
                    float s = out.row(4 + c)[i];
                    if (s > best) { best = s; bc = c; }
                }
                if (best < conf_th) continue;
                Det d; d.cls = bc; d.conf = best;
                d.x1 = (cx - bw / 2) / r; d.y1 = (cy - bh / 2) / r;
                d.x2 = (cx + bw / 2) / r; d.y2 = (cy + bh / 2) / r;
                cand.push_back(d);
            } else {
                col = out.row(i);
                cx = col[0]; cy = col[1]; bw = col[2]; bh = col[3];
                float best = 0; int bc = -1;
                for (int c = 0; c < ncls; c++) {
                    float s = col[4 + c];
                    if (s > best) { best = s; bc = c; }
                }
                if (best < conf_th) continue;
                Det d; d.cls = bc; d.conf = best;
                d.x1 = (cx - bw / 2) / r; d.y1 = (cy - bh / 2) / r;
                d.x2 = (cx + bw / 2) / r; d.y2 = (cy + bh / 2) / r;
                cand.push_back(d);
            }
        }
        // greedy per-class NMS
        std::sort(cand.begin(), cand.end(),
                  [](const Det& a, const Det& b) { return a.conf > b.conf; });
        std::vector<Det> keep;
        std::vector<char> removed(cand.size(), 0);
        for (size_t i = 0; i < cand.size(); i++) {
            if (removed[i]) continue;
            keep.push_back(cand[i]);
            for (size_t j = i + 1; j < cand.size(); j++)
                if (!removed[j] && cand[j].cls == cand[i].cls &&
                    iou(cand[i], cand[j]) > nms_th)
                    removed[j] = 1;
        }
        printf("%zu\n", keep.size());
        for (auto& d : keep)
            printf("%d %.3f %.1f %.1f %.1f %.1f\n",
                   d.cls, d.conf, d.x1, d.y1, d.x2, d.y2);
        fflush(stdout);
    }
    return 0;
}
