#include "pennis.hpp"

VkResult CreateDebugUtilsMessengerEXT(
    VkInstance instance,
    const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
    const VkAllocationCallbacks* pAllocator,
    VkDebugUtilsMessengerEXT* pDebugMessenger)
{
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (func != nullptr)
        return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
    else
        return VK_ERROR_EXTENSION_NOT_PRESENT;
}

void DestroyDebugUtilsMessengerEXT(
    VkInstance instance,
    VkDebugUtilsMessengerEXT debugMessenger,
    const VkAllocationCallbacks* pAllocator)
{
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
    if (func != nullptr)
        func(instance, debugMessenger, pAllocator);
}

PENNIS::PENNIS(const uint32_t batchSize,
               const std::vector<uint32_t>& layerSizes,
               const std::vector<uint32_t>& activationTypes,
               const AdamParams adamParams)
    : batchSize(batchSize), adamParams(adamParams)
{
    size_t numLayers = layerSizes.size() - 1;
    layers.resize(numLayers);
    for (size_t i = 0; i < numLayers; i++)
    {
        layers[i].inSize = layerSizes[i];
        layers[i].outSize = layerSizes[i + 1];
        layers[i].actType = activationTypes[i];
    }
    
    createInstance();
    setupDebugMessenger();
    pickPhysicalDevice();
    createLogicalDevice();
    createComputeDescriptorSetLayout();
    createComputePipeline("shaders/forward.spv",  forwardPipelineLayout,  forwardPipeline);
    createComputePipeline("shaders/backprop.spv", backpropPipelineLayout, backpropPipeline);
    createComputePipeline("shaders/adam.spv",     adamPipelineLayout,     adamPipeline);
    createComputePipeline("shaders/reduce.spv",   reducePipelineLayout,   reducePipeline);
    createCommandPool();
    createShaderStorageBuffers();
    createDescriptorPool();
    createComputeDescriptorSets();
    createComputeCommandBuffers();
    createSyncObjects();
}

PENNIS::~PENNIS()
{
    vkDestroyPipeline(device, forwardPipeline, nullptr);
    vkDestroyPipelineLayout(device, forwardPipelineLayout, nullptr);

    vkDestroyPipeline(device, backpropPipeline, nullptr);
    vkDestroyPipelineLayout(device, backpropPipelineLayout, nullptr);

    vkDestroyPipeline(device, reducePipeline, nullptr);
    vkDestroyPipelineLayout(device, reducePipelineLayout, nullptr);

    vkDestroyPipeline(device, adamPipeline, nullptr);
    vkDestroyPipelineLayout(device, adamPipelineLayout, nullptr);

    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    
    vkDestroyDescriptorSetLayout(device, forwardDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, backpropDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, reduceDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, adamDescriptorSetLayout, nullptr);

    for (auto& layer : layers)
    {
        destroyBuffer(layer.weights);
        destroyBuffer(layer.biases);
        destroyBuffer(layer.input);
        destroyBuffer(layer.preActs);
        destroyBuffer(layer.output);
        destroyBuffer(layer.dWeightsBatch);
        destroyBuffer(layer.dWeights);
        destroyBuffer(layer.dBiasesBatch);
        destroyBuffer(layer.dBiases);
        destroyBuffer(layer.dInput);
        destroyBuffer(layer.delta);
        destroyBuffer(layer.dOutput);
        destroyBuffer(layer.mWeights);
        destroyBuffer(layer.vWeights);
        destroyBuffer(layer.mBiases);
        destroyBuffer(layer.vBiases);
    }

    layers.clear();

    destroyBuffer(targetBuffer);

    vkDestroyFence(device, fence, nullptr);
    
    vkDestroyCommandPool(device, commandPool, nullptr);

    vkDestroyDevice(device, nullptr);

    if (enableValidationLayers)
        DestroyDebugUtilsMessengerEXT(instance, debugMessenger, nullptr);

    vkDestroyInstance(instance, nullptr);
}

void PENNIS::createDeviceLocalBufferWithStaging(const void* srcData, VkDeviceSize size, VkBufferUsageFlags dstUsage, Buffer& dstBuf)
{
    Buffer staging;
    createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 staging);

    void* data;
    vkMapMemory(device, staging.memory, 0, size, 0, &data);
    memcpy(data, srcData, (size_t)size);
    vkUnmapMemory(device, staging.memory);

    createBuffer(size, dstUsage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, dstBuf);

    VkCommandBuffer copyCmd = beginSingleTimeCommands();
    VkBufferCopy copyRegion{};
    copyRegion.srcOffset = 0;
    copyRegion.dstOffset = 0;
    copyRegion.size = size;
    vkCmdCopyBuffer(copyCmd, staging.buffer, dstBuf.buffer, 1, &copyRegion);
    endSingleTimeCommands(copyCmd);

    vkDestroyBuffer(device, staging.buffer, nullptr);
    vkFreeMemory(device, staging.memory, nullptr);
}

void PENNIS::uploadToDeviceBuffer(const void* srcData, VkDeviceSize size, Buffer& dstBuf)
{
    Buffer staging;
    createBuffer(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 staging);

    void* data;
    vkMapMemory(device, staging.memory, 0, size, 0, &data);
    memcpy(data, srcData, (size_t)size);
    vkUnmapMemory(device, staging.memory);

    VkCommandBuffer cmd = beginSingleTimeCommands();
    VkBufferCopy copyRegion{0,0,size};
    vkCmdCopyBuffer(cmd, staging.buffer, dstBuf.buffer, 1, &copyRegion);
    endSingleTimeCommands(cmd);

    vkDestroyBuffer(device, staging.buffer, nullptr);
    vkFreeMemory(device, staging.memory, nullptr);
}

void PENNIS::readbackFromDeviceBuffer(Buffer& srcBuf, void* dstMemory, VkDeviceSize size)
{
    Buffer staging;
    createBuffer(size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 staging);

    VkCommandBuffer cmd = beginSingleTimeCommands();
    VkBufferCopy copyRegion{0,0,size};
    vkCmdCopyBuffer(cmd, srcBuf.buffer, staging.buffer, 1, &copyRegion);
    endSingleTimeCommands(cmd);

    void* data;
    vkMapMemory(device, staging.memory, 0, size, 0, &data);
    memcpy(dstMemory, data, (size_t)size);
    vkUnmapMemory(device, staging.memory);

    vkDestroyBuffer(device, staging.buffer, nullptr);
    vkFreeMemory(device, staging.memory, nullptr);
}

void PENNIS::uploadInputs(const std::vector<float>& inputData)
{
    Layer& inputLayer = layers.front();
    uploadToDeviceBuffer(inputData.data(), inputData.size() * sizeof(float), inputLayer.input);
}

void PENNIS::uploadTargets(const std::vector<float>& targetData)
{
    uploadToDeviceBuffer(targetData.data(), targetData.size() * sizeof(float), targetBuffer);
}

inline VkBufferMemoryBarrier makeBarrier(VkBuffer buf)
{
    return VkBufferMemoryBarrier{
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        nullptr,
        VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED,
        buf,
        0,
        VK_WHOLE_SIZE
    };
}

void PENNIS::train()
{
    vkResetCommandBuffer(computeCommandBuffer, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(computeCommandBuffer, &beginInfo) != VK_SUCCESS)
        throw std::runtime_error("failed to begin recording compute command buffer!");

    recordForwardBatchCommandBuffer();

    {
        std::vector<VkBufferMemoryBarrier> bufBarriers;
        bufBarriers.reserve(layers.size() * 2);
        for (auto &L : layers)
        {
            bufBarriers.push_back(makeBarrier(L.preActs.buffer));
            bufBarriers.push_back(makeBarrier(L.output.buffer));
        }

        vkCmdPipelineBarrier(
            computeCommandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr,
            layers.size() * 2, bufBarriers.data(),
            0, nullptr
        );
    }

    recordBackpropCommandBuffer();

    {
        std::vector<VkBufferMemoryBarrier> bufBarriers;
        bufBarriers.reserve(layers.size() * 2);
        for (auto &L : layers)
        {
            bufBarriers.push_back(makeBarrier(L.dWeightsBatch.buffer));
            bufBarriers.push_back(makeBarrier(L.dBiasesBatch.buffer));
        }

        vkCmdPipelineBarrier(
            computeCommandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr,
            layers.size() * 2, bufBarriers.data(),
            0, nullptr
        );
    }

    recordReduceCommandBuffer();

    {
        std::vector<VkBufferMemoryBarrier> bufBarriers;
        bufBarriers.reserve(layers.size() * 2);
        for (auto &L : layers) {
            bufBarriers.push_back(makeBarrier(L.dWeights.buffer));
            bufBarriers.push_back(makeBarrier(L.dBiases.buffer));
        }

        vkCmdPipelineBarrier(
            computeCommandBuffer,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0, nullptr,
            layers.size() * 2, bufBarriers.data(),
            0, nullptr
        );
    }

    recordAdamCommandBuffer();

    if (vkEndCommandBuffer(computeCommandBuffer) != VK_SUCCESS)
        throw std::runtime_error("failed to end recording compute command buffer!");

    computeSubmission();

    adamTimestep++;
}

void PENNIS::runForward()
{
    vkResetCommandBuffer(computeCommandBuffer, 0);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(computeCommandBuffer, &beginInfo) != VK_SUCCESS)
        throw std::runtime_error("failed to begin recording compute command buffer!");

    recordForwardCommandBuffer();

    if (vkEndCommandBuffer(computeCommandBuffer) != VK_SUCCESS)
        throw std::runtime_error("failed to end recording compute command buffer!");

    computeSubmission();
}

std::vector<float> PENNIS::predict(const std::vector<float>& inputData)
{
    uploadInputs(inputData);
    runForward();

    Layer& outLayer = layers.back();
    uint32_t outputSize = outLayer.outSize;

    std::vector<float> output(outputSize);
    readbackFromDeviceBuffer(outLayer.output, output.data(), outputSize * sizeof(float));

    return output;
}

const char* activationName(uint32_t activationType)
{
    switch (activationType)
    {
        case ReLU: return "ReLU";
        case Sigmoid: return "Sigmoid";
        case Tanh: return "Tanh";
        default: return "None";
    }
}

void PENNIS::printArchitecture()
{
    Layer& outLayer = layers.back();
    uint32_t totalOutputSize = batchSize * outLayer.outSize;

    std::vector<float> output(totalOutputSize);
    readbackFromDeviceBuffer(outLayer.output, output.data(), totalOutputSize * sizeof(float));

    std::vector<float> target(totalOutputSize);
    readbackFromDeviceBuffer(targetBuffer, target.data(), totalOutputSize * sizeof(float));

    float loss = 0.0f;
    #pragma omp parallel for reduction(+:loss)
    for (size_t i = 0; i < totalOutputSize; i++)
    {
        double diff = output[i] - target[i];
        loss += diff * diff;
    }
    loss /= batchSize;

    std::cout << "\nNeural network architecture: (Loss: " << loss << ")\n";

    for (size_t i = 0; i < layers.size(); i++)
    {
        const auto& L = layers[i];

        int32_t inputSize  = L.inSize;
        int32_t outputSize = L.outSize;

        std::cout << "Layer " << i << ": " << inputSize << " -> " << outputSize
                  << " Activation: " << activationName(L.actType);
        std::cout << "\nWeights:\n";

        std::vector<float> weights((size_t)inputSize * outputSize);
        readbackFromDeviceBuffer(const_cast<Buffer&>(L.weights), weights.data(),
                                 (VkDeviceSize)inputSize * outputSize * sizeof(float));

        for (size_t j = 0; j < weights.size(); ++j)
            std::cout << weights[j] << " ";
        
        std::cout << "\nBiases:\n";

        std::vector<float> biases(outputSize);
        readbackFromDeviceBuffer(const_cast<Buffer&>(L.biases), biases.data(),
                                 (VkDeviceSize)outputSize * sizeof(float));

        for (size_t j = 0; j < biases.size(); ++j)
            std::cout << biases[j] << " ";
        std::cout << "\n\n";
    }
}

void PENNIS::createInstance()
{
    if (enableValidationLayers && !checkValidationLayerSupport())
        throw std::runtime_error("validation layers requested, but not available!");

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "PENNIS";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    auto extensions = getRequiredExtensions();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if (enableValidationLayers)
    {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();

        populateDebugMessengerCreateInfo(debugCreateInfo);
        createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*) &debugCreateInfo;
    }

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS)
        throw std::runtime_error("failed to create instance!");
}

void PENNIS::populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo)
{
    createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;
}

void PENNIS::setupDebugMessenger()
{
    if (!enableValidationLayers) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    populateDebugMessengerCreateInfo(createInfo);

    if (CreateDebugUtilsMessengerEXT(instance, &createInfo, nullptr, &debugMessenger) != VK_SUCCESS)
        throw std::runtime_error("failed to set up debug messenger!");
}

void PENNIS::pickPhysicalDevice()
{
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0)
        throw std::runtime_error("failed to find GPUs with Vulkan support!");

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const auto& device : devices)
    {
        if (isDeviceSuitable(device))
        {
            physicalDevice = device;
            break;
        }
    }

    if (physicalDevice == VK_NULL_HANDLE)
        throw std::runtime_error("failed to find a suitable GPU!");
}

void PENNIS::createLogicalDevice()
{
    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = indices.computeFamily.value();
    queueCreateInfo.queueCount = 1;

    float queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceFeatures deviceFeatures{};

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.queueCreateInfoCount = 1;

    createInfo.pEnabledFeatures = &deviceFeatures;

    createInfo.enabledExtensionCount = 0;

    if (enableValidationLayers)
    {
        createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
        createInfo.ppEnabledLayerNames = validationLayers.data();
    }

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS)
        throw std::runtime_error("failed to create logical device!");

    vkGetDeviceQueue(device, indices.computeFamily.value(), 0, &computeQueue);
}

void PENNIS::createComputeDescriptorSetLayout()
{
    std::array<VkDescriptorSetLayoutBinding, 5> forwardBinding = {{
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    }};

    VkDescriptorSetLayoutCreateInfo forwardInfo{};
    forwardInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    forwardInfo.bindingCount = 5;
    forwardInfo.pBindings = forwardBinding.data();

    if (vkCreateDescriptorSetLayout(device, &forwardInfo, nullptr, &forwardDescriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create forward descriptor set layout!");
    
    std::array<VkDescriptorSetLayoutBinding, 10> backpropBinding = {{
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    }};

    VkDescriptorSetLayoutCreateInfo backpropInfo{};
    backpropInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    backpropInfo.bindingCount = 10;
    backpropInfo.pBindings = backpropBinding.data();

    if (vkCreateDescriptorSetLayout(device, &backpropInfo, nullptr, &backpropDescriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create backprop descriptor set layout!");
    
    std::array<VkDescriptorSetLayoutBinding, 2> reduceBinding = {{
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    }};

    VkDescriptorSetLayoutCreateInfo reduceInfo{};
    reduceInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    reduceInfo.bindingCount = 2;
    reduceInfo.pBindings = reduceBinding.data();

    if (vkCreateDescriptorSetLayout(device, &reduceInfo, nullptr, &reduceDescriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create reduce descriptor set layout!");
    
    std::array<VkDescriptorSetLayoutBinding, 4> adamBinding = {{
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    }};

    VkDescriptorSetLayoutCreateInfo adamInfo{};
    adamInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    adamInfo.bindingCount = 4;
    adamInfo.pBindings = adamBinding.data();

    if (vkCreateDescriptorSetLayout(device, &adamInfo, nullptr, &adamDescriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create adam descriptor set layout!");
}

void PENNIS::createComputePipeline(
    const std::string& shaderPath,
    VkPipelineLayout& outPipelineLayout,
    VkPipeline& outPipeline)
{
    auto computeShaderCode = readFile(shaderPath);

    VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);
    
    VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
    computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computeShaderStageInfo.module = computeShaderModule;
    computeShaderStageInfo.pName = "main";

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushRange.offset = 0;

    uint32_t size;
    VkDescriptorSetLayout layout;

    if      (shaderPath.find("forward")  != std::string::npos) size = 4 * sizeof(uint32_t),                      layout = forwardDescriptorSetLayout;
    else if (shaderPath.find("backprop") != std::string::npos) size = 6 * sizeof(uint32_t),                      layout = backpropDescriptorSetLayout;
    else if (shaderPath.find("reduce")   != std::string::npos) size = 2 * sizeof(uint32_t),                      layout = reduceDescriptorSetLayout;
    else                                                       size = 2 * sizeof(uint32_t) + sizeof(AdamParams), layout = adamDescriptorSetLayout;

    pushRange.size = size;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &layout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushRange;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &outPipelineLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create compute pipeline layout!");

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = outPipelineLayout;
    pipelineInfo.stage = computeShaderStageInfo;

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &outPipeline) != VK_SUCCESS)
        throw std::runtime_error("failed to create compute pipeline!");

    vkDestroyShaderModule(device, computeShaderModule, nullptr);
}

void PENNIS::createCommandPool()
{
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamilyIndices.computeFamily.value();

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create command pool!");
    }
}

void PENNIS::createShaderStorageBuffers()
{
    size_t numLayers = layers.size();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < numLayers; i++)
    {
        Layer& L = layers[i];

        uint32_t inSize  = L.inSize;
        uint32_t outSize = L.outSize;

        VkDeviceSize weightsSize = (VkDeviceSize)inSize * (VkDeviceSize)outSize * sizeof(float);
        VkDeviceSize biasesSize  = (VkDeviceSize)outSize * sizeof(float);
        VkDeviceSize inputSize   = (VkDeviceSize)batchSize * (VkDeviceSize)inSize * sizeof(float);
        VkDeviceSize preActsSize = (VkDeviceSize)batchSize * (VkDeviceSize)outSize * sizeof(float);
        VkDeviceSize outputSize  = (VkDeviceSize)batchSize * (VkDeviceSize)outSize * sizeof(float);
        VkDeviceSize dInputSize  = (VkDeviceSize)batchSize * (VkDeviceSize)inSize * sizeof(float);
        VkDeviceSize dOutputSize = (VkDeviceSize)batchSize * (VkDeviceSize)outSize * sizeof(float);
        VkDeviceSize dWBatchSize = (VkDeviceSize)batchSize * (VkDeviceSize)inSize * (VkDeviceSize)outSize * sizeof(float);
        VkDeviceSize dWSize      = weightsSize;
        VkDeviceSize dBBatchSize = (VkDeviceSize)batchSize * (VkDeviceSize)outSize * sizeof(float);
        VkDeviceSize dBSize      = biasesSize;
        VkDeviceSize mWSize      = weightsSize;
        VkDeviceSize vWSize      = weightsSize;
        VkDeviceSize mBSize      = biasesSize;
        VkDeviceSize vBSize      = biasesSize;

        std::vector<float> initWeights((size_t)inSize * outSize);
        for (size_t j = 0; j < initWeights.size(); ++j) initWeights[j] = dist(gen);

        createDeviceLocalBufferWithStaging(initWeights.data(), weightsSize,
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                           L.weights);

        std::vector<float> initBiases((size_t)outSize);
        for (size_t j = 0; j < initBiases.size(); ++j) initBiases[j] = dist(gen);

        createDeviceLocalBufferWithStaging(initBiases.data(), biasesSize,
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                           L.biases);

        createBuffer(inputSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     L.input);

        createBuffer(preActsSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     L.preActs);

        createBuffer(outputSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     L.output);

        createBuffer(dInputSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     L.dInput);

        createBuffer(dOutputSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     L.delta);

        createBuffer(dOutputSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     L.dOutput);

        createBuffer(dWBatchSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     L.dWeightsBatch);

        createBuffer(dWSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     L.dWeights);

        createBuffer(dBBatchSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     L.dBiasesBatch);

        createBuffer(dBSize,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                     L.dBiases);

        std::vector<float> zerosW((size_t)inSize * outSize, 0.0f);
        createDeviceLocalBufferWithStaging(zerosW.data(), mWSize,
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           L.mWeights);
        createDeviceLocalBufferWithStaging(zerosW.data(), vWSize,
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           L.vWeights);

        std::vector<float> zerosB((size_t)outSize, 0.0f);
        createDeviceLocalBufferWithStaging(zerosB.data(), mBSize,
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           L.mBiases);
        createDeviceLocalBufferWithStaging(zerosB.data(), vBSize,
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           L.vBiases);
    }

    uint32_t targetSize = batchSize * layers.back().outSize;
    createBuffer((VkDeviceSize)targetSize * sizeof(float),
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 targetBuffer);
}

void PENNIS::createDescriptorPool()
{
    uint32_t numLayers = static_cast<uint32_t>(layers.size());
    uint32_t numDescriptorSets = numLayers * 6;

    uint32_t totalDescriptors = numLayers * 27;

    VkDescriptorPoolSize poolSize{};
    poolSize.type            = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = totalDescriptors;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes    = &poolSize;
    poolInfo.maxSets       = numDescriptorSets;

    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
        throw std::runtime_error("failed to create descriptor pool!");
}


void PENNIS::createComputeDescriptorSets()
{
    uint32_t numLayers = static_cast<uint32_t>(layers.size());
    
    std::vector<VkDescriptorSetLayout> forwardLayouts(numLayers, forwardDescriptorSetLayout);
    std::vector<VkDescriptorSetLayout> backpropLayouts(numLayers, backpropDescriptorSetLayout);
    std::vector<VkDescriptorSetLayout> reduceLayouts(2 * numLayers, reduceDescriptorSetLayout);
    std::vector<VkDescriptorSetLayout> adamLayouts(2 * numLayers, adamDescriptorSetLayout);

    {
        VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocInfo.descriptorPool     = descriptorPool;
        allocInfo.descriptorSetCount = numLayers;
        allocInfo.pSetLayouts        = forwardLayouts.data();
        forwardDescriptorSets.resize(numLayers);
        if (vkAllocateDescriptorSets(device, &allocInfo, forwardDescriptorSets.data()) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate forward descriptor sets!");
    }

    {
        VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocInfo.descriptorPool     = descriptorPool;
        allocInfo.descriptorSetCount = numLayers;
        allocInfo.pSetLayouts        = backpropLayouts.data();
        backpropDescriptorSets.resize(numLayers);
        if (vkAllocateDescriptorSets(device, &allocInfo, backpropDescriptorSets.data()) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate backprop descriptor sets!");
    }

    {
        VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocInfo.descriptorPool     = descriptorPool;
        allocInfo.descriptorSetCount = 2 * numLayers;
        allocInfo.pSetLayouts        = reduceLayouts.data();
        reduceDescriptorSets.resize(2 * numLayers);
        if (vkAllocateDescriptorSets(device, &allocInfo, reduceDescriptorSets.data()) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate reduce descriptor sets!");
    }

    {
        VkDescriptorSetAllocateInfo allocInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocInfo.descriptorPool     = descriptorPool;
        allocInfo.descriptorSetCount = 2 * numLayers;
        allocInfo.pSetLayouts        = adamLayouts.data();
        adamDescriptorSets.resize(2 * numLayers);
        if (vkAllocateDescriptorSets(device, &allocInfo, adamDescriptorSets.data()) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate Adam descriptor sets!");
    }

    for (size_t i = 0; i < layers.size(); i++)
    {
        Layer& L = layers[i];

        auto w = bufferInfo(L.weights);
        auto b = bufferInfo(L.biases);
        auto x = bufferInfo(i == 0 ? L.input : layers[i - 1].output);
        auto z = bufferInfo(L.preActs);
        auto a = bufferInfo(L.output);
        auto dx = bufferInfo(L.dInput);
        auto dz = bufferInfo(L.delta);
        auto da = bufferInfo(i == layers.size() - 1 ? L.dOutput : layers[i + 1].dInput);
        auto dWb = bufferInfo(L.dWeightsBatch);
        auto dW  = bufferInfo(L.dWeights);
        auto dBb = bufferInfo(L.dBiasesBatch);
        auto dB  = bufferInfo(L.dBiases);
        auto mW = bufferInfo(L.mWeights);
        auto vW = bufferInfo(L.vWeights);
        auto mB = bufferInfo(L.mBiases);
        auto vB = bufferInfo(L.vBiases);
        auto T  = bufferInfo(targetBuffer);

        std::array<VkWriteDescriptorSet, 5> fwdWrites = {{
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, forwardDescriptorSets[i], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &w, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, forwardDescriptorSets[i], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &b, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, forwardDescriptorSets[i], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &x, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, forwardDescriptorSets[i], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &z, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, forwardDescriptorSets[i], 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &a, nullptr}
        }};
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(fwdWrites.size()), fwdWrites.data(), 0, nullptr);

        std::array<VkWriteDescriptorSet, 10> backWrites = {{
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, backpropDescriptorSets[i], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &w, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, backpropDescriptorSets[i], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &x, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, backpropDescriptorSets[i], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &z, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, backpropDescriptorSets[i], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &a, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, backpropDescriptorSets[i], 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dWb, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, backpropDescriptorSets[i], 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dBb, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, backpropDescriptorSets[i], 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dx, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, backpropDescriptorSets[i], 7, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dz, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, backpropDescriptorSets[i], 8, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &da, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, backpropDescriptorSets[i], 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &T,  nullptr}
        }};
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(backWrites.size()), backWrites.data(), 0, nullptr);

        std::array<VkWriteDescriptorSet, 2> reduceWrites1 = {{
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, reduceDescriptorSets[i * 2], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dWb, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, reduceDescriptorSets[i * 2], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dW,  nullptr}
        }};
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(reduceWrites1.size()), reduceWrites1.data(), 0, nullptr);

        std::array<VkWriteDescriptorSet, 2> reduceWrites2 = {{
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, reduceDescriptorSets[i * 2 + 1], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dBb, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, reduceDescriptorSets[i * 2 + 1], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dB,  nullptr}
        }};
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(reduceWrites2.size()), reduceWrites2.data(), 0, nullptr);

        std::array<VkWriteDescriptorSet, 4> adamWrites1 = {{
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, adamDescriptorSets[i * 2], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &w,  nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, adamDescriptorSets[i * 2], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dW, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, adamDescriptorSets[i * 2], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &mW, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, adamDescriptorSets[i * 2], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &vW, nullptr}
        }};
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(adamWrites1.size()), adamWrites1.data(), 0, nullptr);

        std::array<VkWriteDescriptorSet, 4> adamWrites2 = {{
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, adamDescriptorSets[i * 2 + 1], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &b,  nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, adamDescriptorSets[i * 2 + 1], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dB, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, adamDescriptorSets[i * 2 + 1], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &mB, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, adamDescriptorSets[i * 2 + 1], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &vB, nullptr}
        }};
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(adamWrites2.size()), adamWrites2.data(), 0, nullptr);
    }
}

void PENNIS::createBuffer(VkDeviceSize size,
                          VkBufferUsageFlags usage,
                          VkMemoryPropertyFlags properties,
                          Buffer& buf)
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buf.buffer) != VK_SUCCESS)
        throw std::runtime_error("failed to create buffer!");

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buf.buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &buf.memory) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate buffer memory!");

    vkBindBufferMemory(device, buf.buffer, buf.memory, 0);
}

VkCommandBuffer PENNIS::beginSingleTimeCommands()
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = commandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    return commandBuffer;
}

void PENNIS::endSingleTimeCommands(VkCommandBuffer commandBuffer)
{
    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(computeQueue);

    vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
}

uint32_t PENNIS::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void PENNIS::createComputeCommandBuffers()
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    if (vkAllocateCommandBuffers(device, &allocInfo, &computeCommandBuffer) != VK_SUCCESS)
        throw std::runtime_error("failed to allocate compute command buffers!");
}

void PENNIS::createSyncObjects()
{
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = 0;

    if (vkCreateFence(device, &fenceInfo, nullptr, &fence) != VK_SUCCESS)
        throw std::runtime_error("failed to create fence for compute submission");
}

void PENNIS::recordForwardCommandBuffer()
{
    vkCmdBindPipeline(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, forwardPipeline);

    for (size_t i = 0; i < layers.size(); ++i)
    {
        Layer& L = layers[i];

        vkCmdBindDescriptorSets(
            computeCommandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            forwardPipelineLayout,
            0, 1, &forwardDescriptorSets[i],
            0, nullptr);

        struct Push { uint32_t inSize, outSize, batchSize, actType; };
        Push push = { L.inSize, L.outSize, 1, L.actType };
        vkCmdPushConstants(computeCommandBuffer, forwardPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

        uint32_t groups = (L.outSize - 1) / 256 + 1;
        vkCmdDispatch(computeCommandBuffer, groups, 1, 1);

        if (i + 1 < layers.size())
        {
            auto memBarrier = makeBarrier(L.output.buffer);
            vkCmdPipelineBarrier(
                computeCommandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr,
                1, &memBarrier,
                0, nullptr
            );
        }
    }
}

void PENNIS::recordForwardBatchCommandBuffer()
{
    vkCmdBindPipeline(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, forwardPipeline);

    for (size_t i = 0; i < layers.size(); ++i)
    {
        Layer& L = layers[i];

        vkCmdBindDescriptorSets(
            computeCommandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            forwardPipelineLayout,
            0, 1, &forwardDescriptorSets[i],
            0, nullptr);

        struct Push { uint32_t inSize, outSize, batchSize, actType; };
        Push push = { L.inSize, L.outSize, batchSize, L.actType };
        vkCmdPushConstants(computeCommandBuffer, forwardPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

        uint32_t groups = (L.outSize * batchSize - 1) / 256 + 1;
        vkCmdDispatch(computeCommandBuffer, groups, 1, 1);

        if (i + 1 < layers.size())
        {
            auto memBarrier = makeBarrier(L.output.buffer);
            vkCmdPipelineBarrier(
                computeCommandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr,
                1, &memBarrier,
                0, nullptr
            );
        }
    }
}

void PENNIS::recordBackpropCommandBuffer()
{
    vkCmdBindPipeline(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, backpropPipeline);

    bool isOutput = true;

    struct Push { uint32_t inSize, outSize, batchSize, actType, isOutput, phase; } push;

    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; --i)
    {
        Layer& L = layers[i];

        vkCmdBindDescriptorSets(
            computeCommandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            backpropPipelineLayout,
            0, 1,
            &backpropDescriptorSets[i],
            0, nullptr);

        push = { L.inSize, L.outSize, batchSize, L.actType, isOutput ? 1u : 0u, 0u };
        vkCmdPushConstants(computeCommandBuffer, backpropPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

        uint32_t groups = (L.outSize * batchSize - 1) / 256 + 1;
        vkCmdDispatch(computeCommandBuffer, groups, 1, 1);

        {
            auto memBarrier = makeBarrier(L.delta.buffer);

            vkCmdPipelineBarrier(
                computeCommandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr,
                1, &memBarrier,
                0, nullptr
            );
        }

        push.phase = 1u;
        vkCmdPushConstants(computeCommandBuffer, backpropPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

        groups = (L.inSize * batchSize - 1) / 256 + 1;
        vkCmdDispatch(computeCommandBuffer, groups, 1, 1);

        if (i > 0)
        {
            auto memBarrier = makeBarrier(L.dInput.buffer);

            vkCmdPipelineBarrier(
                computeCommandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr,
                1, &memBarrier,
                0, nullptr
            );
        }

        if (isOutput) isOutput = false;
    }
}

void PENNIS::recordReduceCommandBuffer()
{
    vkCmdBindPipeline(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, reducePipeline);

    struct Push { uint32_t size, batchSize; } push;
    push.batchSize = batchSize;

    uint32_t groups;

    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--)
    {
        Layer& L = layers[i];

        vkCmdBindDescriptorSets(
            computeCommandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            reducePipelineLayout,
            0, 1,
            &reduceDescriptorSets[i * 2],
            0, nullptr);

        push.size = L.inSize * L.outSize;
        vkCmdPushConstants(computeCommandBuffer, reducePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

        groups = (L.inSize * L.outSize - 1) / 256 + 1;
        vkCmdDispatch(computeCommandBuffer, groups, 1, 1);

        vkCmdBindDescriptorSets(
            computeCommandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            reducePipelineLayout,
            0, 1,
            &reduceDescriptorSets[i * 2 + 1],
            0, nullptr);

        push.size = L.outSize;
        vkCmdPushConstants(computeCommandBuffer, reducePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

        groups = (L.outSize - 1) / 256 + 1;
        vkCmdDispatch(computeCommandBuffer, groups, 1, 1);

        {
            VkMemoryBarrier memBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
            memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(
                computeCommandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                1, &memBarrier, 0, nullptr, 0, nullptr);
        }
    }
}

void PENNIS::recordAdamCommandBuffer()
{
    vkCmdBindPipeline(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, adamPipeline);

    struct Push { AdamParams adamParams; uint32_t t, size; } push;
    push.adamParams = adamParams;
    push.t          = adamTimestep;

    uint32_t groups;

    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--)
    {
        Layer& L = layers[i];

        vkCmdBindDescriptorSets(
            computeCommandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            adamPipelineLayout,
            0, 1,
            &adamDescriptorSets[i * 2],
            0, nullptr);

        push.size = L.inSize * L.outSize;
        vkCmdPushConstants(computeCommandBuffer, adamPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

        groups = (L.inSize * L.outSize - 1) / 256 + 1;
        vkCmdDispatch(computeCommandBuffer, groups, 1, 1);

        vkCmdBindDescriptorSets(
            computeCommandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            adamPipelineLayout,
            0, 1,
            &adamDescriptorSets[i * 2 + 1],
            0, nullptr);

        push.size = L.outSize;
        vkCmdPushConstants(computeCommandBuffer, adamPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

        groups = (L.outSize - 1) / 256 + 1;
        vkCmdDispatch(computeCommandBuffer, groups, 1, 1);

        {
            VkMemoryBarrier memBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
            memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(
                computeCommandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                1, &memBarrier, 0, nullptr, 0, nullptr);
        }
    }
}

void PENNIS::computeSubmission()
{
    vkResetFences(device, 1, &fence);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &computeCommandBuffer;

    submitInfo.signalSemaphoreCount = 0;
    submitInfo.pSignalSemaphores = nullptr;

    if (vkQueueSubmit(computeQueue, 1, &submitInfo, fence) != VK_SUCCESS)
        throw std::runtime_error("failed to submit compute command buffer!");
    
    vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX);
}