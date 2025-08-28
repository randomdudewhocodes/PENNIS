#include <zlib.h>
#include "pennis.hpp"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <cfloat>
#include <algorithm>
#include <numeric>
#include <random>
#include <omp.h>

bool readMNISTImages(const std::string &filename, std::vector<std::vector<uint8_t>> &images, int &numImages, int &rows, int &cols)
{
    gzFile file = gzopen(filename.c_str(), "rb");
    if (!file)
    {
        std::cerr << "Cannot open file: " << filename << "\n";
        return false;
    }

    uint32_t magic = 0, nImages = 0, nRows = 0, nCols = 0;

    if (gzread(file, &magic, 4) != 4 || gzread(file, &nImages, 4) != 4 || gzread(file, &nRows, 4) != 4 || gzread(file, &nCols, 4) != 4) {
        std::cerr << "Failed reading header from: " << filename << "\n";
        gzclose(file);
        return false;
    }

    auto swapEndian = [](uint32_t val)
    {
        return ((val >> 24) & 0xFF) | ((val >> 8) & 0xFF00) |
               ((val << 8) & 0xFF0000) | ((val << 24) & 0xFF000000);
    };

    magic = swapEndian(magic);
    numImages = swapEndian(nImages);
    rows = swapEndian(nRows);
    cols = swapEndian(nCols);

    if (magic != 2051)
    {
        std::cerr << "Invalid MNIST image file magic for " << filename << " (got " << magic << ")\n";
        gzclose(file);
        return false;
    }

    images.clear();
    images.resize(numImages, std::vector<uint8_t>(rows * cols));
    for (int i = 0; i < numImages; ++i) {
        int toRead = rows * cols;
        int got = gzread(file, images[i].data(), toRead);
        if (got != toRead) {
            std::cerr << "Failed to read image " << i << " from " << filename << " (got " << got << " bytes)\n";
            gzclose(file);
            return false;
        }
    }

    gzclose(file);
    return true;
}

bool readMNISTLabels(const std::string &filename, std::vector<uint8_t> &labels, int &numLabels)
{
    gzFile file = gzopen(filename.c_str(), "rb");
    if (!file) {
        std::cerr << "Cannot open labels file: " << filename << "\n";
        return false;
    }

    uint32_t magic = 0, nLabels = 0;
    if (gzread(file, &magic, 4) != 4 || gzread(file, &nLabels, 4) != 4) {
        std::cerr << "Failed reading header from labels file: " << filename << "\n";
        gzclose(file);
        return false;
    }

    auto swapEndian = [](uint32_t val) {
        return ((val >> 24) & 0xFF) | ((val >> 8) & 0xFF00) |
               ((val << 8) & 0xFF0000) | ((val << 24) & 0xFF000000);
    };

    magic = swapEndian(magic);
    numLabels = swapEndian(nLabels);

    if (magic != 2049)
    {
        std::cerr << "Invalid MNIST label file magic for " << filename << " (got " << magic << ")\n";
        gzclose(file);
        return false;
    }

    labels.clear();
    labels.resize(numLabels);
    int got = gzread(file, labels.data(), numLabels);
    if (got != numLabels) {
        std::cerr << "Failed to read labels from " << filename << " (got " << got << " bytes)\n";
        gzclose(file);
        return false;
    }

    gzclose(file);
    return true;
}

inline float lineSegmentSDF(float px, float py,
                            float ax, float ay,
                            float bx, float by)
{
    float vx = px - ax;
    float vy = py - ay;
    float ux = bx - ax;
    float uy = by - ay;
    float len2 = ux*ux + uy*uy;
    float t = (len2 > 0.0f) ? (vx*ux + vy*uy) / len2 : 0.0f;
    t = std::clamp(t, 0.0f, 1.0f);
    float dx = vx - t*ux;
    float dy = vy - t*uy;
    return std::sqrt(dx*dx + dy*dy);
}

static inline uint8_t sample_bilinear(const std::vector<uint8_t>& img, int rows, int cols, float x, float y)
{
    if (x < 0.0f || y < 0.0f || x > cols - 1 || y > rows - 1) return 0;
    int x0 = int(std::floor(x));
    int y0 = int(std::floor(y));
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float wx = x - float(x0);
    float wy = y - float(y0);

    auto get = [&](int xx, int yy) -> uint8_t {
        if (xx < 0 || yy < 0 || xx >= cols || yy >= rows) return 0;
        return img[size_t(yy) * cols + xx];
    };

    float v00 = float(get(x0, y0));
    float v10 = float(get(x1, y0));
    float v01 = float(get(x0, y1));
    float v11 = float(get(x1, y1));

    float v0 = v00 * (1.0f - wx) + v10 * wx;
    float v1 = v01 * (1.0f - wx) + v11 * wx;
    float v  = v0  * (1.0f - wy) + v1  * wy;
    int iv = int(std::round(v));
    if (iv < 0) iv = 0;
    if (iv > 255) iv = 255;
    return (uint8_t)iv;
}

static std::vector<uint8_t> augmentImageAffine(const std::vector<uint8_t>& src, int rows, int cols, float tx, float ty, float scale, float angleDeg)
{
    std::vector<uint8_t> dst(size_t(rows) * cols, 0);
    const float cx = (cols - 1) * 0.5f;
    const float cy = (rows - 1) * 0.5f;
    const float rad = angleDeg * 3.14159265358979323846f / 180.0f;
    const float c = std::cos(rad);
    const float s = std::sin(rad);

    // parallelize rows with OpenMP. each thread writes distinct dst rows so this is safe.
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            float dx = float(x);
            float dy = float(y);
            float rx = dx - cx - tx;
            float ry = dy - cy - ty;
            float sx = ( c * rx + s * ry) / scale + cx;
            float sy = (-s * rx + c * ry) / scale + cy;
            dst[size_t(y) * cols + x] = sample_bilinear(src, rows, cols, sx, sy);
        }
    }
    return dst;
}

// ------------------ end augmentation helpers ------------------

int main()
{
    try
    {
        std::vector<std::vector<uint8_t>> trainImages, testImages;
        std::vector<uint8_t> trainLabels, testLabels;
        int numTrainImages = 0, numTestImages = 0, rows = 0, cols = 0;

        if (!readMNISTImages("src/examples/mnist training demo/train-images-idx3-ubyte.gz", trainImages, numTrainImages, rows, cols)) return -1;
        if (!readMNISTLabels("src/examples/mnist training demo/train-labels-idx1-ubyte.gz", trainLabels, numTrainImages)) return -1;
        if (!readMNISTImages("src/examples/mnist training demo/t10k-images-idx3-ubyte.gz", testImages, numTestImages, rows, cols)) return -1;
        if (!readMNISTLabels("src/examples/mnist training demo/t10k-labels-idx1-ubyte.gz", testLabels, numTestImages)) return -1;

        std::vector<uint32_t> layerSizes = {uint32_t(rows * cols), 64, 64, 10};
        std::vector<uint32_t> actTypes = {Tanh, Tanh, Sigmoid};
        AdamParams adamParams = {0.9f, 0.999f, 1e-8f, 0.001f, 0.01f};

        const int batchSize = 600;
        PENNIS pennis(256, batchSize, layerSizes, actTypes, adamParams);

        const int epochs = 30000;
        int currentEpoch = 0;
        bool saved = false;

        std::vector<float> trainInput, trainTarget;
        std::vector<float> testInput;

        const int aug_per_image = 14;

        size_t totalSamples = size_t(numTrainImages) * (1 + aug_per_image);

        trainInput.clear();
        trainTarget.clear();
        trainInput.resize(totalSamples * size_t(rows) * cols);
        trainTarget.resize(totalSamples * 10);

        std::atomic<size_t> progressCounter(0);
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < numTrainImages; ++i)
        {
            std::mt19937 rng_local(i);
            std::uniform_real_distribution<float> tdist_local(-5.0f, 5.0f);
            std::uniform_real_distribution<float> sdist_local(-0.2f, 0.2f);
            std::uniform_real_distribution<float> rdist_local(-2.0f, 2.0f);
            std::uniform_real_distribution<float> gdist_local(-0.7f, 0.7f);

            for (int a = 0; a <= aug_per_image; ++a)
            {
                size_t sampleIndex = size_t(i) * (1 + aug_per_image) + size_t(a);
                size_t inputOffset = sampleIndex * size_t(rows) * cols;
                size_t targetOffset = sampleIndex * 10;
                size_t pixelCount = size_t(rows) * cols;

                if (a == 0)
                {
                    for (int j = 0; j < rows * cols; ++j)
                        trainInput[inputOffset + j] = float(trainImages[i][j]) / 255.0f;
                }
                else
                {
                    float tx = tdist_local(rng_local);
                    float ty = tdist_local(rng_local);
                    float scale = sdist_local(rng_local);
                    float rot = rdist_local(rng_local);
                    float gamma = gdist_local(rng_local);

                    std::vector<uint8_t> aug = augmentImageAffine(trainImages[i], rows, cols, tx, ty, exp(scale), rot);
                    for (int j = 0; j < rows * cols; ++j)
                        trainInput[inputOffset + j] = pow(float(aug[j]) / 255.0f, exp(gamma));
                }

                for (int k = 0; k < 10; ++k)
                    trainTarget[targetOffset + k] = (trainLabels[i] == k) ? 1.0f : 0.0f;
                
                size_t done = ++progressCounter;
                if (done % 5000 == 0 || done == totalSamples) {
                    #pragma omp critical
                    {
                        std::cout << "Loaded " << done << "/" << totalSamples << " images\n";
                    }
                }
            }
        }

        std::cout << "All training images loaded." << std::endl;

        pennis.uploadTrainInputs(trainInput);
        pennis.uploadTrainTargets(trainTarget);

        testInput.reserve(size_t(numTestImages) * rows * cols);
        for (int i = 0; i < numTestImages; i++)
        {
            for (int j = 0; j < rows * cols; j++)
                testInput.push_back(float(testImages[i][j]) / 255.0f);
        }

        if (!glfwInit())
        {
            std::cerr << "glfwInit() failed\n";
            return -1;
        }

        GLFWwindow* window = glfwCreateWindow(1000, 700, "NN Training Visualizer / Live Demo", NULL, NULL);
        if (!window)
        {
            std::cerr << "glfwCreateWindow() returned NULL\n";
            glfwTerminate();
            return -1;
        }

        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 130");

        float loss = 0.0f;
        float successRate = 0.0f;
        std::vector<float> successArray;

        enum Mode { MODE_TRAIN = 0, MODE_DEMO = 1 };
        Mode mode = MODE_TRAIN;

        const int demoSize = 28;
        std::vector<float> demoPixels((size_t)demoSize * demoSize, 0.0f);
        int predictedDigit = -1;
        std::vector<float> lastProbs(10, 0.0f);

        float brush_radius = 1.25f;
        float brush_strength = 0.9f;

        ImVec2 prevDrawPos(-1.0f, -1.0f);

        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImVec2 windowSize = ImGui::GetIO().DisplaySize;
            float fontSize = ImGui::GetFontSize();

            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(windowSize);

            ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar |
                                     ImGuiWindowFlags_NoResize |
                                     ImGuiWindowFlags_NoMove |
                                     ImGuiWindowFlags_NoScrollbar |
                                     ImGuiWindowFlags_NoCollapse |
                                     ImGuiWindowFlags_NoSavedSettings;

            ImGui::Begin("MNIST - Train / Live demo", nullptr, flags);
            if (ImGui::BeginTabBar("ModeTabs", ImGuiTabBarFlags_None))
            {
                if (ImGui::BeginTabItem("Train")) { mode = MODE_TRAIN; ImGui::EndTabItem(); }
                if (ImGui::BeginTabItem("Live demo")) { mode = MODE_DEMO; ImGui::EndTabItem(); }
                ImGui::EndTabBar();
            }

            if (mode == MODE_TRAIN)
            {
                ImGui::Text("Epoch: %d / %d", currentEpoch, epochs);
                ImGui::Text("Loss: %f", loss);
                ImGui::Text("Success Rate: %.2f%%", successRate * 100.0f);

                ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();
                ImVec2 canvas_sz = ImGui::GetContentRegionAvail();
                ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);
                ImDrawList* draw_list = ImGui::GetWindowDrawList();
                draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(240, 240, 240, 255));

                const int ticks = 5;
                for (int t = 0; t <= ticks; ++t)
                {
                    float y = canvas_p0.y + canvas_sz.y * (1.0f - float(t) / ticks);
                    draw_list->AddLine(ImVec2(canvas_p0.x, y), ImVec2(canvas_p1.x, y), IM_COL32(200,200,200,120));
                }

                size_t n = successArray.size();
                for (size_t i = 0; i < n; ++i)
                {
                    if(i > 0)
                    {
                        float x0 = canvas_p0.x + float(i)     / float(n) * canvas_sz.x;
                        float x1 = canvas_p0.x + float(i + 1) / float(n) * canvas_sz.x;
                        float y0 = canvas_p0.y + canvas_sz.y * (1.0f - successArray[i - 1]);
                        float y1 = canvas_p0.y + canvas_sz.y * (1.0f - successArray[i]);
                        draw_list->AddLine(ImVec2(x0, y0), ImVec2(x1, y1), IM_COL32(0, 0, 255, 255), 2.0f);
                    }
                    else
                    {
                        float x0 = canvas_p0.x;
                        float x1 = canvas_p0.x + canvas_sz.x / float(n);
                        float y0 = canvas_p0.y + canvas_sz.y * (1.0f - 0.);
                        float y1 = canvas_p0.y + canvas_sz.y * (1.0f - successArray[0]);
                        draw_list->AddLine(ImVec2(x0, y0), ImVec2(x1, y1), IM_COL32(0, 0, 255, 255), 2.0f);
                    }
                }

                ImGui::Dummy(canvas_sz);

                if(!saved)
                {
                    if (currentEpoch >= epochs)
                    {
                        pennis.saveArchitecture("src/examples/trained models/mnist_model.pnn");
                        std::cout << "Training complete. Saved model.\n";
                        saved = true;
                    }

                    if(currentEpoch > 0 && currentEpoch % 50 == 0)
                    {
                        std::vector<float> allOutputs = pennis.predict(testInput);

                        if ((int)allOutputs.size() >= numTestImages * 10)
                        {
                            int correct = 0;
                            for (int i = 0; i < numTestImages; i++)
                            {
                                int number = 0;
                                float maxOutput = allOutputs[i * 10 + 0];
                                for (int j = 1; j < 10; j++)
                                {
                                    float v = allOutputs[i * 10 + j];
                                    if (v > maxOutput)
                                    {
                                        maxOutput = v;
                                        number = j;
                                    }
                                }
                                if (number == testLabels[i]) ++correct;
                            }
                            successRate = float(correct) / float(std::max(1, numTestImages));
                            successArray.push_back(successRate);
                        }
                        else
                        {
                            std::cerr << "predict() returned unexpected output size: " << allOutputs.size() << "\n";
                        }
                    }
                }

                if (currentEpoch < epochs && !saved)
                {
                    pennis.train();
                    currentEpoch++;
                    loss = pennis.getLoss();
                }
            }
            else
            {
                ImGui::Text("Draw a digit (left click paint, right click erase). Then press Predict.");
                ImGui::Separator();

                ImVec2 avail = ImGui::GetContentRegionAvail();
                float total_w = avail.x;
                float total_h = avail.y;

                float canvas_side = std::min(total_w * 0.65f, total_h);
                float right_w = total_w - canvas_side - 10.0f;

                if (right_w < 180.0f) right_w = 180.0f;

                ImGui::BeginChild("demo_left", ImVec2(canvas_side, canvas_side), false, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
                ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();
                ImVec2 canvas_sz = ImVec2(canvas_side, canvas_side);
                ImGui::InvisibleButton("demo_canvas", canvas_sz);
                ImDrawList* draw_list = ImGui::GetWindowDrawList();

                ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);
                draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(255,255,255,255));
                float cell = canvas_sz.x / float(demoSize);
                ImGuiIO& io = ImGui::GetIO();
                bool hovered = ImGui::IsItemHovered();
                bool mouseDownL = io.MouseDown[0];
                bool mouseDownR = io.MouseDown[1];

                if (hovered && (mouseDownL || mouseDownR))
                {
                    ImVec2 mpos = io.MousePos;
                    float local_x = mpos.x - canvas_p0.x;
                    float local_y = mpos.y - canvas_p0.y;
                    float fx = local_x / cell;
                    float fy = local_y / cell;

                    ImVec2 curPos(fx, fy);
                    ImVec2 startPos = curPos;
                    if (prevDrawPos.x >= 0.0f && prevDrawPos.y >= 0.0f && (mouseDownL || mouseDownR))
                    {
                        int y0 = std::max(0, int(std::floor(std::min(prevDrawPos.y, curPos.y) - brush_radius)));
                        int y1 = std::min(demoSize - 1, int(std::ceil(std::max(prevDrawPos.y, curPos.y) + brush_radius)));
                        int x0 = std::max(0, int(std::floor(std::min(prevDrawPos.x, curPos.x) - brush_radius)));
                        int x1 = std::min(demoSize - 1, int(std::ceil(std::max(prevDrawPos.x, curPos.x) + brush_radius)));

                        for (int yy = y0; yy <= y1; ++yy)
                        for (int xx = x0; xx <= x1; ++xx)
                        {
                            float cx = (float)xx + 0.5f;
                            float cy = (float)yy + 0.5f;

                            float distp = lineSegmentSDF(cx, cy,
                                                        prevDrawPos.x, prevDrawPos.y,
                                                        curPos.x, curPos.y);

                            if (distp > brush_radius) continue;

                            float w = 1.0f - distp / brush_radius;
                            w = w*w*(3.0f - 2.0f*w); // smooth falloff
                            float sign = mouseDownL ? 1.0f : -1.0f;
                            float dt60 = (io.DeltaTime > 0.0f) ? (io.DeltaTime * 60.0f) : 1.0f;
                            float delta = brush_strength * w * sign * dt60;

                            size_t idx = size_t(yy) * demoSize + size_t(xx);
                            demoPixels[idx] = std::clamp(demoPixels[idx] + delta, 0.0f, 1.0f);
                        }
                    }
                    else
                    {
                        // Fallback: single point stroke when no prevDrawPos yet
                        int y0 = std::max(0, int(std::floor(fy - brush_radius)));
                        int y1 = std::min(demoSize - 1, int(std::ceil(fy + brush_radius)));
                        int x0 = std::max(0, int(std::floor(fx - brush_radius)));
                        int x1 = std::min(demoSize - 1, int(std::ceil(fx + brush_radius)));

                        for (int yy = y0; yy <= y1; ++yy)
                        for (int xx = x0; xx <= x1; ++xx)
                        {
                            float cx = (float)xx + 0.5f;
                            float cy = (float)yy + 0.5f;
                            float dx = cx - fx;
                            float dy = cy - fy;
                            float distp = std::sqrt(dx*dx + dy*dy);
                            if (distp > brush_radius) continue;

                            float w = 1.0f - distp / brush_radius;
                            w = w*w*(3.0f - 2.0f*w);
                            float sign = mouseDownL ? 1.0f : -1.0f;
                            float dt60 = (io.DeltaTime > 0.0f) ? (io.DeltaTime * 60.0f) : 1.0f;
                            float delta = brush_strength * w * sign * dt60;

                            size_t idx = size_t(yy) * demoSize + size_t(xx);
                            demoPixels[idx] = std::clamp(demoPixels[idx] + delta, 0.0f, 1.0f);
                        }
                    }

                    prevDrawPos = curPos;
                }
                else
                {
                    prevDrawPos = ImVec2(-1.0f, -1.0f);
                }

                for (int y = 0; y < demoSize; ++y)
                {
                    for (int x = 0; x < demoSize; ++x)
                    {
                        size_t idx = (size_t)y * demoSize + (size_t)x;
                        float v = demoPixels[idx];
                        unsigned int c = (unsigned int)std::round(255.0f * v);
                        ImU32 col = IM_COL32(c, c, c, 255);
                        ImVec2 p0 = ImVec2(canvas_p0.x + x * cell, canvas_p0.y + y * cell);
                        ImVec2 p1 = ImVec2(p0.x + cell, p0.y + cell);
                        draw_list->AddRectFilled(p0, p1, col);
                    }
                }

                for (int i = 0; i <= demoSize; ++i)
                {
                    float xi = canvas_p0.x + i * cell;
                    float yi = canvas_p0.y + i * cell;
                    draw_list->AddLine(ImVec2(canvas_p0.x, yi), ImVec2(canvas_p1.x, yi), IM_COL32(200,200,200,30));
                    draw_list->AddLine(ImVec2(xi, canvas_p0.y), ImVec2(xi, canvas_p1.y), IM_COL32(200,200,200,30));
                }

                ImGui::Dummy(canvas_sz);
                ImGui::EndChild();

                ImGui::SameLine();

                ImGui::BeginChild("demo_right", ImVec2(right_w, canvas_side), false, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
                ImGui::Text("Controls");
                ImGui::Separator();

                ImGui::SliderFloat("Brush radius (cells)", &brush_radius, 0.25f, 4.0f, "%.2f");
                ImGui::SliderFloat("Brush strength", &brush_strength, 0.05f, 2.0f, "%.2f");
                ImGui::TextWrapped("Tip: hold left mouse to paint, right mouse to erase.");
                ImGui::Separator();

                if (ImGui::Button("Clear", ImVec2(-1, 0))) {
                    std::fill(demoPixels.begin(), demoPixels.end(), 0.0f);
                    predictedDigit = -1;
                    std::fill(lastProbs.begin(), lastProbs.end(), 0.0f);
                }

                std::vector<float> inp;
                inp.reserve(demoSize * demoSize);
                for (size_t i = 0; i < demoPixels.size(); ++i)
                    inp.push_back(demoPixels[i]);

                try {
                    std::vector<float> probs = pennis.predict(inp);
                    if (probs.size() >= 10) {
                        lastProbs = probs;
                        int best = 0;
                        for (int k = 1; k < 10; ++k) if (probs[k] > probs[best]) best = k;
                        predictedDigit = best;
                    } else {
                        std::cerr << "predict() returned unexpected size: " << probs.size() << "\n";
                    }
                } catch (const std::exception &e) {
                    std::cerr << "Exception in predict(): " << e.what() << "\n";
                }

                ImGui::Spacing();
                ImGui::Separator();
                ImGui::Spacing();

                ImGui::Text("Prediction");
                ImGui::Dummy(ImVec2(0,4));
                if (predictedDigit >= 0) {
                    ImGui::Text("Predicted: %d", predictedDigit);
                } else {
                    ImGui::Text("Predicted: -");
                }
                ImGui::Dummy(ImVec2(0,6));

                for (int k = 0; k < 10; ++k)
                {
                    char buf[32];
                    snprintf(buf, sizeof(buf), "%d: %.2f", k, (k < (int)lastProbs.size() ? lastProbs[k] : 0.0f));
                    float frac = (k < (int)lastProbs.size() ? lastProbs[k] : 0.0f);
                    ImGui::ProgressBar(frac, ImVec2(-1, 0), buf);
                    ImGui::Spacing();
                }

                ImGui::EndChild();
            }

            ImGui::End();

            ImGui::Render();
            int display_w, display_h;
            glfwGetFramebufferSize(window, &display_w, &display_h);
            glViewport(0, 0, display_w, display_h);
            glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            glfwSwapBuffers(window);
        }

        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        glfwDestroyWindow(window);
        glfwTerminate();

        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Unhandled exception: " << e.what() << std::endl;
        std::cerr << "Aborting.\n";
        return -1;
    }
    catch (...)
    {
        std::cerr << "Unhandled non-standard exception\n";
        return -2;
    }
}
