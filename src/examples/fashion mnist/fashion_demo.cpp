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

int main()
{
    try
    {
        std::vector<std::vector<uint8_t>> trainImages, testImages;
        std::vector<uint8_t> trainLabels, testLabels;
        int numTrainImages = 0, numTestImages = 0, rows = 0, cols = 0;

        if (!readMNISTImages("src/examples/fashion mnist/train-images-idx3-ubyte.gz", trainImages, numTrainImages, rows, cols)) return -1;
        if (!readMNISTLabels("src/examples/fashion mnist/train-labels-idx1-ubyte.gz", trainLabels, numTrainImages)) return -1;
        if (!readMNISTImages("src/examples/fashion mnist/t10k-images-idx3-ubyte.gz", testImages, numTestImages, rows, cols)) return -1;
        if (!readMNISTLabels("src/examples/fashion mnist/t10k-labels-idx1-ubyte.gz", testLabels, numTestImages)) return -1;

        std::vector<uint32_t> layerSizes = {uint32_t(rows * cols), 256, 128, 64, 10};
        std::vector<uint32_t> actTypes = {ReLU, ReLU, ReLU, Sigmoid};
        AdamParams adamParams = {0.9f, 0.999f, 1e-8f, 0.001f, 0.01f};

        const int batchSize = 600;
        PENNIS pennis(256, batchSize, layerSizes, actTypes, adamParams);

        const int epochs = 30000;
        int currentEpoch = 0;
        bool saved = false;

        std::vector<float> trainInput, trainTarget;
        std::vector<float> testInput;

        const int aug_per_image = 14;

        size_t totalSamples = size_t(numTrainImages);

        trainInput.clear();
        trainTarget.clear();
        trainInput.resize(totalSamples * size_t(rows) * cols);
        trainTarget.resize(totalSamples * 10);

        std::atomic<size_t> progressCounter(0);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numTrainImages; ++i)
        {
            const float inv255 = 1.0f / 255.0f;

            #pragma omp simd
            for (int j = 0; j < rows * cols; ++j)
                trainInput[i * rows * cols + j] = float(trainImages[i][j]) * inv255;
            
            for (int k = 0; k < 10; ++k)
                trainTarget[i * 10 + k] = float(trainLabels[i] == k);
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

        bool predsComputed = false;
        std::vector<int> predictedLabels;         // predicted label per test image
        std::vector<std::vector<float>> allProbs; // per-image probs (optional)
        std::vector<int> successIndices;
        std::vector<int> failIndices;
        enum ShowList { SHOW_NONE = 0, SHOW_SUCCESS = 1, SHOW_FAIL = 2 };
        ShowList currentShow = SHOW_NONE;
        int currentListPos = 0;

        // <<< CHANGED: Fashion-MNIST class names
        const char* fashionLabels[10] = {
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
        };

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

            // <<< CHANGED: Window title mentions Fashion-MNIST
            ImGui::Begin("Fashion-MNIST - Train / Live demo", nullptr, flags);
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
                        // <<< CHANGED: save model under a different name (optional)
                        pennis.saveArchitecture("src/examples/trained models/fashion_model.pnn");
                        std::cout << "Training complete. Saved model." << std::endl;
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
                // --- NEW: layout matched to reference demo ---
                ImGui::Text("Demo viewer — browse success / failure cases from the test set.");
                ImGui::Separator();

                if (!predsComputed)
                {
                    try {
                        std::vector<float> flatOutputs = pennis.predict(testInput);
                        if ((int)flatOutputs.size() >= numTestImages * 10) {
                            predictedLabels.resize(numTestImages);
                            allProbs.clear();
                            allProbs.resize(numTestImages, std::vector<float>(10, 0.0f));
                            successIndices.clear();
                            failIndices.clear();

                            for (int i = 0; i < numTestImages; ++i) {
                                int best = 0;
                                float bestv = flatOutputs[i * 10 + 0];
                                for (int k = 0; k < 10; ++k) {
                                    float v = flatOutputs[i * 10 + k];
                                    allProbs[i][k] = v;                     // <<< FIX: store all probs
                                    if (k == 0 || v > bestv) { bestv = v; best = k; }
                                }
                                predictedLabels[i] = best;
                                if (best == (int)testLabels[i]) successIndices.push_back(i);
                                else failIndices.push_back(i);
                            }
                            predsComputed = true;
                            currentShow = SHOW_NONE;
                            currentListPos = 0;
                            std::cout << "Predictions computed for demo (" << numTestImages << " images). successes: " << successIndices.size() << ", failures: " << failIndices.size() << "\n";
                        } else {
                            std::cerr << "predict() returned unexpected output size in demo: " << flatOutputs.size() << "\n";
                        }
                    } catch (const std::exception &e) {
                        std::cerr << "Exception computing predictions for demo: " << e.what() << "\n";
                    }
                }

                ImVec2 avail = ImGui::GetContentRegionAvail();
                float total_w = avail.x;
                float total_h = avail.y;

                // left canvas takes ~65% width, right controls take rest (like reference)
                float canvas_side = std::min(total_w * 0.65f, total_h);
                float right_w = total_w - canvas_side - 10.0f;
                if (right_w < 180.0f) right_w = 180.0f;

                // Left: image canvas
                ImGui::BeginChild("demo_left", ImVec2(canvas_side, canvas_side), false, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
                ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();
                ImVec2 canvas_sz = ImVec2(canvas_side, canvas_side);
                ImDrawList* draw_list = ImGui::GetWindowDrawList();
                ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);
                draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(0,0,0,255));

                // find which list is active
                const std::vector<int>* activeList = nullptr;
                const char* listName = "None";
                if (currentShow == SHOW_SUCCESS) { activeList = &successIndices; listName = "Successes"; }
                else if (currentShow == SHOW_FAIL) { activeList = &failIndices; listName = "Failures"; }

                if (!activeList) {
                    // draw a placeholder grid of the currently selected index (none)
                    draw_list->AddText(ImVec2(canvas_p0.x + 8, canvas_p0.y + 8), IM_COL32(80,80,80,255), "No list selected. Use the controls on the right.");
                } else if (activeList->empty()) {
                    draw_list->AddText(ImVec2(canvas_p0.x + 8, canvas_p0.y + 8), IM_COL32(80,80,80,255), "Selected list is empty.");
                } else {
                    int imgIdx = (*activeList)[currentListPos];
                    float cell = canvas_sz.x / float(cols);
                    for (int y = 0; y < rows; ++y)
                    {
                        for (int x = 0; x < cols; ++x)
                        {
                            size_t idx = size_t(y) * cols + size_t(x);
                            uint8_t v = testImages[imgIdx][idx];
                            float fv = float(v) / 255.0f;
                            int c = int(std::round(255.0f * fv));
                            ImU32 col = IM_COL32(c, c, c, 255);
                            ImVec2 p0 = ImVec2(canvas_p0.x + x * cell, canvas_p0.y + y * cell);
                            ImVec2 p1 = ImVec2(p0.x + cell, p0.y + cell);
                            draw_list->AddRectFilled(p0, p1, col);
                        }
                    }
                }

                ImGui::Dummy(canvas_sz);
                ImGui::EndChild();

                ImGui::SameLine();

                // Right: controls and progress bars
                ImGui::BeginChild("demo_right", ImVec2(right_w, canvas_side), false, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

                // Buttons to pick which list to view (moved to right column to match reference)
                if (ImGui::Button("Show Success Cases", ImVec2(-1, 0))) {
                    currentShow = SHOW_SUCCESS;
                    currentListPos = 0;
                }
                ImGui::Spacing();
                if (ImGui::Button("Show Failure Cases", ImVec2(-1, 0))) {
                    currentShow = SHOW_FAIL;
                    currentListPos = 0;
                }
                ImGui::Spacing();
                if (ImGui::Button("Clear Selection", ImVec2(-1, 0))) {
                    currentShow = SHOW_NONE;
                }

                ImGui::Spacing();

                // decide which vector to use (again for right column context)
                if (!activeList) {
                    ImGui::Text("No list selected.");
                } else if (activeList->empty()) {
                    ImGui::Text("%s list is empty.", listName);
                } else {
                    ImGui::Text("%s: %zu items. Showing %d / %zu", listName, activeList->size(), currentListPos + 1, activeList->size());

                    // navigation buttons in one row (like reference)
                    if (ImGui::Button("Prev")) {
                        if (currentListPos > 0) --currentListPos;
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Next")) {
                        if ((size_t)currentListPos + 1 < activeList->size()) ++currentListPos;
                    }
                    ImGui::SameLine();
                    if (ImGui::Button("Go to 1st")) currentListPos = 0;
                    ImGui::SameLine();
                    if (ImGui::Button("Go to Last")) currentListPos = (int)activeList->size() - 1;

                    ImGui::Spacing();

                    int imgIdx = (*activeList)[currentListPos];
                    int predicted = predictedLabels.empty() ? -1 : predictedLabels[imgIdx];
                    int actual = testLabels[imgIdx];
                    if (predicted >= 0) {
                        ImGui::Text("Predicted: %s (%d)", (predicted >=0 && predicted < 10) ? fashionLabels[predicted] : "?", predicted);
                        ImGui::SameLine();
                        ImGui::Text("   Actual: %s (%d)", (actual >=0 && actual < 10) ? fashionLabels[actual] : "?", actual);
                    } else {
                        ImGui::Text("Predicted: -    Actual: %d", actual);
                    }

                    ImGui::Separator();

                    // Show per-class progress bars with numbers (full-width, like the reference)
                    for (int k = 0; k < 10; ++k)
                    {
                        float frac = (k < (int)allProbs[imgIdx].size() ? allProbs[imgIdx][k] : 0.0f);
                        char buf[128];
                        snprintf(buf, sizeof(buf), "%s: %.4f", fashionLabels[k], frac);
                        ImGui::ProgressBar(frac, ImVec2(-1, 0), buf);
                        ImGui::Spacing();
                    }
                }

                ImGui::Separator();
                ImGui::Text("Note: predictions are computed from the current model state at demo entry.");
                ImGui::Text("If you retrain more, switch tabs or restart the demo to recompute.");

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
