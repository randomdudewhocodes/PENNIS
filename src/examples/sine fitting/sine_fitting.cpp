#include "pennis.hpp"
#include "imgui.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_opengl3.h"
#include <GLFW/glfw3.h>
#include <vector>
#include <cmath>
#include <iostream>

int main()
{
    try
    {
        std::vector<uint32_t> layerSizes = {1, 16, 16, 1};
        std::vector<uint32_t> actTypes = {Tanh, Tanh, None};
        AdamParams adamParams = {0.9f, 0.999f, 1e-8f, 0.005f, 0.01f};

        int numSamples = 50;
        PENNIS pennis(256, numSamples, layerSizes, actTypes, adamParams);

        const int epochs = 5000;
        int currentEpoch = 0;
        bool saved = false;

        std::vector<float> trainInput, trainTarget;

        for (int i = 0; i < numSamples; i++)
        {
            float x = float(i) / (numSamples - 1) * 2 * 3.141593f;
            trainInput.push_back(x);
            trainTarget.push_back(sinf(x));
        }

        pennis.uploadTrainInputs(trainInput);
        pennis.uploadTrainTargets(trainTarget);

        if (!glfwInit())
        {
            std::cerr << "glfwInit() failed\n";
            return -1;
        }

        GLFWwindow* window = glfwCreateWindow(800, 600, "NN Training Visualizer", NULL, NULL);
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

        std::vector<float> nn(numSamples);

        float loss = 0.0f;

        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            if (currentEpoch == epochs && !saved)
            {
                pennis.saveArchitecture("src/examples/trained models/sine_model.pnn");
                std::cout << "Training complete. Saved model.\n";
                saved = true;
            }

            std::vector<float> nn = pennis.predict(trainInput);

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

            ImGui::Begin("NN Training Visualizer", nullptr, flags);
            ImGui::Text("Epoch: %d / %d", currentEpoch, epochs);
            ImGui::Text("Loss: %f", loss);

            ImVec2 canvas_p0 = ImGui::GetCursorScreenPos();
            ImVec2 canvas_sz = ImGui::GetContentRegionAvail();
            ImVec2 canvas_p1 = ImVec2(canvas_p0.x + canvas_sz.x, canvas_p0.y + canvas_sz.y);

            ImDrawList* draw_list = ImGui::GetWindowDrawList();
            draw_list->AddRectFilled(canvas_p0, canvas_p1, IM_COL32(240, 240, 240, 255));

            float midY = canvas_p0.y + canvas_sz.y / 2;
            draw_list -> AddLine(ImVec2(canvas_p0.x, midY), ImVec2(canvas_p1.x, midY), IM_COL32(150, 150, 150, 255));

            float scale = canvas_sz.y / 4.0f;

            for (int px = 1; px < numSamples; px++)
            {
                ImVec2 p0 = ImVec2(canvas_p0.x + trainInput[px - 1] / (2*3.141593f) * canvas_sz.x,
                                   midY - trainTarget[px - 1] * scale);
                ImVec2 p1 = ImVec2(canvas_p0.x + trainInput[px] / (2*3.141593f) * canvas_sz.x,
                                   midY - trainTarget[px] * scale);
                draw_list -> AddLine(p0, p1, IM_COL32(0, 0, 255, 255), 2.0f);
            }

            for (int px = 1; px < numSamples; px++)
            {
                ImVec2 p0 = ImVec2(canvas_p0.x + trainInput[px - 1] / (2*3.141593f) * canvas_sz.x,
                                   midY - nn[px - 1] * scale);
                ImVec2 p1 = ImVec2(canvas_p0.x + trainInput[px] / (2*3.141593f) * canvas_sz.x,
                                   midY - nn[px] * scale);
                draw_list -> AddLine(p0, p1, IM_COL32(255, 0, 0, 255), 2.0f);
            }

            ImGui::Dummy(canvas_sz);
            ImGui::End();

            for (int i = 0; i < 50 && currentEpoch < epochs; i++)
            {
                pennis.train();
                currentEpoch++;
            }
            
            if (!saved) loss = pennis.getLoss();

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