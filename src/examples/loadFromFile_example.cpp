// demo_from_file.cpp
// simple demo that loads a saved .pnn model using PENNIS::loadFromFile
// and runs the "Live demo" drawing/predict UI similar to mnist_fitting.cpp.
//
// Requires: imgui, imgui_impl_glfw, imgui_impl_opengl3, GLFW, Vulkan runtime
// and the compiled pennis.* (pennis.hpp / pennis.cpp) in your project.

#include "pennis.hpp"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <stdexcept>

int main()
{
    try
    {
        std::vector<uint32_t> dummySizes = {1, 1};
        std::vector<uint32_t> dummyActs  = {Sigmoid};
        AdamParams adamParams;

        const uint32_t workgroupSize = 256;
        const uint32_t batchSize = 1;

        PENNIS dummy(workgroupSize, batchSize, dummySizes, dummyActs, adamParams);

        const std::string modelPath = "src/examples/mnist_model.pnn";

        PENNIS* net = nullptr;
        try {
            net = dummy.loadFromFile(modelPath);
        } catch (const std::exception& e) {
            std::cerr << "failed to load model from '" << modelPath << "': " << e.what() << "\n";
            return -1;
        }

        if (!glfwInit()) {
            std::cerr << "glfwInit() failed\n";
            delete net;
            return -1;
        }

        GLFWwindow* window = glfwCreateWindow(900, 650, "Demo - loadFromFile", NULL, NULL);
        if (!window) {
            std::cerr << "glfwCreateWindow() returned NULL\n";
            glfwTerminate();
            delete net;
            return -1;
        }

        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGui::StyleColorsDark();
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init("#version 130");

        const int demoSize = 28;
        std::vector<float> demoPixels((size_t)demoSize * demoSize, 0.0f);
        int predictedDigit = -1;
        std::vector<float> lastProbs(10, 0.0f);

        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            ImVec2 windowSize = ImGui::GetIO().DisplaySize;

            ImGui::SetNextWindowPos(ImVec2(0, 0));
            ImGui::SetNextWindowSize(windowSize);

            ImGuiWindowFlags flags = ImGuiWindowFlags_NoTitleBar |
                                     ImGuiWindowFlags_NoResize |
                                     ImGuiWindowFlags_NoMove |
                                     ImGuiWindowFlags_NoScrollbar |
                                     ImGuiWindowFlags_NoCollapse |
                                     ImGuiWindowFlags_NoSavedSettings;

            ImGui::Begin("Demo - loadFromFile", nullptr, flags);

            ImGui::Text("Draw a digit (left click paint, right click erase). Then press Predict.");
            ImGui::Separator();

            ImVec2 avail = ImGui::GetContentRegionAvail();
            float total_w = avail.x;
            float total_h = avail.y;

            float canvas_side = std::min(total_w * 0.65f, total_h);
            float right_w = total_w - canvas_side - 10.0f;
            if (right_w < 180.0f) right_w = 180.0f;

            ImGui::BeginChild("demo_left", ImVec2(canvas_side, canvas_side), false,
                              ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
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
                int ix = int(local_x / cell);
                int iy = int(local_y / cell);
                if (ix >= 0 && ix < demoSize && iy >= 0 && iy < demoSize)
                {
                    for (int dy = -1; dy <= 1; ++dy)
                        for (int dx = -1; dx <= 1; ++dx)
                        {
                            int nx = ix + dx;
                            int ny = iy + dy;
                            if (nx >= 0 && nx < demoSize && ny >= 0 && ny < demoSize)
                            {
                                size_t idx = (size_t)ny * demoSize + (size_t)nx;
                                if (mouseDownL)
                                    demoPixels[idx] = std::min(1.0f, demoPixels[idx] + 0.35f);
                                else if (mouseDownR)
                                    demoPixels[idx] = std::max(0.0f, demoPixels[idx] - 0.35f);
                            }
                        }
                }
            }

            for (int y = 0; y < demoSize; ++y)
            {
                for (int x = 0; x < demoSize; ++x)
                {
                    size_t idx = (size_t)y * demoSize + (size_t)x;
                    float v = demoPixels[idx]; // 0..1
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

            ImGui::BeginChild("demo_right", ImVec2(right_w, canvas_side), false,
                              ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
            ImGui::Text("Controls");
            ImGui::Separator();

            if (ImGui::Button("Clear", ImVec2(-1, 0))) {
                std::fill(demoPixels.begin(), demoPixels.end(), 0.0f);
                predictedDigit = -1;
                std::fill(lastProbs.begin(), lastProbs.end(), 0.0f);
            }

            if (ImGui::Button("Predict", ImVec2(-1, 0)))
            {
                std::vector<float> inp;
                inp.reserve(demoSize * demoSize);
                for (size_t i = 0; i < demoPixels.size(); ++i) inp.push_back(demoPixels[i]);

                try {
                    std::vector<float> probs = net->predict(inp);
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

        delete net;
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Unhandled exception: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cerr << "Unhandled non-standard exception\n";
        return -2;
    }
}
