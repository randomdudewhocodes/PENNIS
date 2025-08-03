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

PENNIS::PENNIS(const uint32_t workgroupSize,
               const std::vector<uint32_t>& layerSizes,
               const std::vector<uint32_t>& activationTypes,
               const AdamParams adamParams)
    : workgroupSize(workgroupSize), adamParams(adamParams)
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

    vkDestroyPipeline(device, adamPipeline, nullptr);
    vkDestroyPipelineLayout(device, adamPipelineLayout, nullptr);

    vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    
    vkDestroyDescriptorSetLayout(device, forwardDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, backpropDescriptorSetLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, adamDescriptorSetLayout, nullptr);

    for (auto& layer : layers)
    {
        destroyBuffer(layer.weights);
        destroyBuffer(layer.biases);
        destroyBuffer(layer.input);
        destroyBuffer(layer.preActs);
        destroyBuffer(layer.output);
        destroyBuffer(layer.dWeights);
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

void PENNIS::uploadInputs(const std::vector<float>& inputData)
{
    Layer& inputLayer = layers.front();

    void* data;
    vkMapMemory(device, inputLayer.input.memory, 0, inputData.size() * sizeof(float), 0, &data);
    memcpy(data, inputData.data(), inputData.size() * sizeof(float));
    vkUnmapMemory(device, inputLayer.input.memory);
}

void PENNIS::uploadTargets(const std::vector<float>& targetData)
{
    void* data;
    vkMapMemory(device, targetBuffer.memory, 0, targetData.size() * sizeof(float), 0, &data);
    memcpy(data, targetData.data(), targetData.size() * sizeof(float));
    vkUnmapMemory(device, targetBuffer.memory);
}

void PENNIS::runForward()
{
    vkResetCommandBuffer(computeCommandBuffer, 0);
    recordForwardCommandBuffer();
    computeSubmission();
}

void PENNIS::runBackprop()
{
    vkResetCommandBuffer(computeCommandBuffer, 0);
    recordBackpropCommandBuffer();
    computeSubmission();
}

void PENNIS::applyAdam()
{
    vkResetCommandBuffer(computeCommandBuffer, 0);
    recordAdamCommandBuffer();
    computeSubmission();
    adamTimestep++;
}

std::vector<float> PENNIS::predict(const std::vector<float>& inputData)
{
    uploadInputs(inputData);
    runForward();
    Layer& outLayer = layers.back();
    uint32_t outputSize = outLayer.outSize;
    
    std::vector<float> output(outputSize);
    void* data;
    vkMapMemory(device, outLayer.output.memory, 0, outputSize * sizeof(float), 0, &data);
    std::memcpy(output.data(), data, outputSize * sizeof(float));
    vkUnmapMemory(device, outLayer.output.memory);

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
    uint32_t outputSize = outLayer.outSize;
    
    std::vector<float> output(outputSize);
    void* data;
    vkMapMemory(device, outLayer.output.memory, 0, outputSize * sizeof(float), 0, &data);
    std::memcpy(output.data(), data, outputSize * sizeof(float));
    vkUnmapMemory(device, outLayer.output.memory);

    std::vector<float> target(outputSize);
    vkMapMemory(device, targetBuffer.memory, 0, outputSize * sizeof(float), 0, &data);
    std::memcpy(target.data(), data, outputSize * sizeof(float));
    vkUnmapMemory(device, targetBuffer.memory);

    float loss = 0.0f;

    for (size_t i = 0; i < layers.back().outSize; i++)
    {
        double diff = output[i] - target[i];
        loss += diff * diff;
    }

    std::cout << "\nNeural network architecture: (Loss: " << loss << ")\n";

    for (size_t i = 0; i < layers.size(); i++)
    {
        const auto& L = layers[i];

        int32_t inputSize  = L.inSize;
        int32_t outputSize = L.outSize;

        std::cout << "Layer " << i << ": " << inputSize << " -> " << outputSize << " Activation: " << activationName(L.actType);
        std::cout << "\nWeights:\n";

        std::vector<float> weights(inputSize * outputSize);
        std::vector<float> biases(outputSize);

        void* data;
        vkMapMemory(device, L.weights.memory, 0, inputSize * outputSize * sizeof(float), 0, &data);
        std::memcpy(weights.data(), data, inputSize * outputSize * sizeof(float));
        for(int j = 0; j < inputSize * outputSize; j++)
        {
            std::cout << weights[j] << " "; 
        }
        vkUnmapMemory(device, L.weights.memory);
        
        std::cout << "\nBiases:\n";
        vkMapMemory(device, L.biases.memory, 0, outputSize * sizeof(float), 0, &data);
        std::memcpy(biases.data(), data, outputSize * sizeof(float));
        for(int j = 0; j < outputSize; j++)
        {
            std::cout << biases[j] << " "; 
        }
        vkUnmapMemory(device, L.biases.memory);
        
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
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        {14, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    }};

    VkDescriptorSetLayoutCreateInfo backpropInfo{};
    backpropInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    backpropInfo.bindingCount = 10;
    backpropInfo.pBindings = backpropBinding.data();

    if (vkCreateDescriptorSetLayout(device, &backpropInfo, nullptr, &backpropDescriptorSetLayout) != VK_SUCCESS)
        throw std::runtime_error("failed to create backprop descriptor set layout!");
    
    std::array<VkDescriptorSetLayoutBinding, 8> adamBinding = {{
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        {10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        {11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        {12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        {13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
    }};

    VkDescriptorSetLayoutCreateInfo adamInfo{};
    adamInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    adamInfo.bindingCount = 8;
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

    if      (shaderPath.find("forward") != std::string::npos)  size = 3 * sizeof(uint32_t),                      layout = forwardDescriptorSetLayout;
    else if (shaderPath.find("backprop") != std::string::npos) size = 5 * sizeof(uint32_t),                      layout = backpropDescriptorSetLayout;
    else                                                       size = 3 * sizeof(uint32_t) + sizeof(AdamParams), layout = adamDescriptorSetLayout;

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
    
    float* data;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-1, 1);

    for (size_t i = 0; i < numLayers; i++)
    {
        Layer& L = layers[i];
        
        uint32_t inSize = L.inSize;
        uint32_t outSize = L.outSize;

        createBuffer(inSize * outSize * sizeof(float),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     L.weights);
        
        createBuffer(outSize * sizeof(float),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     L.biases);
        
        vkMapMemory(device, L.weights.memory, 0, VK_WHOLE_SIZE, 0, (void**)&data);
        for (uint32_t j = 0; j < inSize * outSize; j++) data[j] = dist(gen);
        vkUnmapMemory(device, L.weights.memory);

        vkMapMemory(device, L.biases.memory, 0, VK_WHOLE_SIZE, 0, (void**)&data);
        for (uint32_t j = 0; j < outSize; j++) data[j] = dist(gen);
        vkUnmapMemory(device, L.biases.memory);

        createBuffer(inSize * sizeof(float),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     L.input);

        createBuffer(outSize * sizeof(float),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     L.preActs);
        
        createBuffer(outSize * sizeof(float),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     L.output);

        createBuffer(inSize * sizeof(float),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     L.dInput);
        
        createBuffer(outSize * sizeof(float),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     L.delta);

        createBuffer(outSize * sizeof(float),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     L.dOutput);

        createBuffer(inSize * outSize * sizeof(float),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     L.dWeights);
        
        createBuffer(outSize * sizeof(float),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                     L.dBiases);
        
        auto allocBuffer = [&](Buffer& buf, size_t size)
        {
            createBuffer(size * sizeof(float),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                         buf);
            
            vkMapMemory(device, buf.memory, 0, VK_WHOLE_SIZE, 0, (void**)&data);
            for (uint32_t j = 0; j < size; j++) data[j] = 0.0f; 
            vkUnmapMemory(device, buf.memory);
        };

        allocBuffer(L.mWeights, inSize * outSize);
        allocBuffer(L.vWeights, inSize * outSize);
        allocBuffer(L.mBiases, outSize);
        allocBuffer(L.vBiases, outSize);
    }

    uint32_t targetSize = layers.back().outSize;
    createBuffer(targetSize * sizeof(float),
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                 targetBuffer);
}

void PENNIS::createDescriptorPool()
{
    uint32_t numLayers = static_cast<uint32_t>(layers.size());
    uint32_t numDescriptorSets = numLayers * 3;

    uint32_t totalDescriptors = numLayers * 23;

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
    std::vector<VkDescriptorSetLayout> adamLayouts(numLayers, adamDescriptorSetLayout);

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
        allocInfo.descriptorSetCount = numLayers;
        allocInfo.pSetLayouts        = adamLayouts.data();
        adamDescriptorSets.resize(numLayers);
        if (vkAllocateDescriptorSets(device, &allocInfo, adamDescriptorSets.data()) != VK_SUCCESS)
            throw std::runtime_error("failed to allocate Adam descriptor sets!");
    }

    for (size_t i = 0; i < layers.size(); i++)
    {
        Layer& L = layers[i];

        auto w = bufferInfo(L.weights);
        auto b = bufferInfo(L.biases);
        auto x = bufferInfo(L.input);
        auto z = bufferInfo(L.preActs);
        auto a = bufferInfo(L.output);
        auto dx = bufferInfo(L.dInput);
        auto dz = bufferInfo(L.delta);
        auto da = bufferInfo(L.dOutput);
        auto dW = bufferInfo(L.dWeights);
        auto dB = bufferInfo(L.dBiases);
        auto mW = bufferInfo(L.mWeights);
        auto vW = bufferInfo(L.vWeights);
        auto mB = bufferInfo(L.mBiases);
        auto vB = bufferInfo(L.vBiases);
        auto T  = bufferInfo(targetBuffer);

        std::array<VkWriteDescriptorSet,5> fwdWrites = {{
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, forwardDescriptorSets[i], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &w, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, forwardDescriptorSets[i], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &b, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, forwardDescriptorSets[i], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &x, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, forwardDescriptorSets[i], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &z, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, forwardDescriptorSets[i], 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &a, nullptr}
        }};
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(fwdWrites.size()), fwdWrites.data(), 0, nullptr);

        std::array<VkWriteDescriptorSet,10> backWrites = {{
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, backpropDescriptorSets[i], 0,  0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &w, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, backpropDescriptorSets[i], 2,  0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &x, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, backpropDescriptorSets[i], 3,  0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &z, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, backpropDescriptorSets[i], 4,  0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &a, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, backpropDescriptorSets[i], 5,  0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dx, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, backpropDescriptorSets[i], 6,  0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dz, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, backpropDescriptorSets[i], 7,  0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &da, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, backpropDescriptorSets[i], 8,  0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dW, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, backpropDescriptorSets[i], 9,  0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dB, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, backpropDescriptorSets[i], 14, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &T,  nullptr}
        }};
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(backWrites.size()), backWrites.data(), 0, nullptr);

        std::array<VkWriteDescriptorSet,8> adamWrites = {{
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, adamDescriptorSets[i], 0,  0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &w,  nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, adamDescriptorSets[i], 1,  0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &b,  nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, adamDescriptorSets[i], 8,  0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dW, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, adamDescriptorSets[i], 9,  0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dB, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, adamDescriptorSets[i], 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &mW, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, adamDescriptorSets[i], 11, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &vW, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, adamDescriptorSets[i], 12, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &mB, nullptr},
            {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, adamDescriptorSets[i], 13, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &vB, nullptr}
        }};
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(adamWrites.size()), adamWrites.data(), 0, nullptr);
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
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(computeCommandBuffer, &beginInfo) != VK_SUCCESS)
        throw std::runtime_error("failed to begin recording compute command buffer!");

    vkCmdBindPipeline(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, forwardPipeline);

    for (int i = 0; i < layers.size(); i++)
    {
        Layer& L = layers[i];

        vkCmdBindDescriptorSets(
            computeCommandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            forwardPipelineLayout,
            0, 1, &forwardDescriptorSets[i],
            0, nullptr);

        struct Push{ uint32_t inSize, outSize, actType; } push = { L.inSize, L.outSize, L.actType };
        vkCmdPushConstants(computeCommandBuffer, forwardPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

        uint32_t groups = (L.outSize - 1) / workgroupSize + 1;
        vkCmdDispatch(computeCommandBuffer, groups, 1, 1);

        if (i < layers.size() - 1)
        {
            VkMemoryBarrier memBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
            memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(
                computeCommandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                1, &memBarrier,
                0, nullptr,
                0, nullptr
            );
            
            VkBuffer src = L.output.buffer,
                     dst = layers[i + 1].input.buffer;
            
            if (src == VK_NULL_HANDLE || dst == VK_NULL_HANDLE) continue;

            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = 0;
            copyRegion.dstOffset = 0;
            copyRegion.size = L.outSize * sizeof(float);

            vkCmdCopyBuffer(computeCommandBuffer, src, dst, 1, &copyRegion);

            VkBufferMemoryBarrier bufBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            bufBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            bufBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            bufBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bufBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bufBarrier.buffer = dst;
            bufBarrier.offset = 0;
            bufBarrier.size = VK_WHOLE_SIZE;

            vkCmdPipelineBarrier(
                computeCommandBuffer,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0, nullptr,
                1, &bufBarrier,
                0, nullptr
            );
        }
    }

    if (vkEndCommandBuffer(computeCommandBuffer) != VK_SUCCESS)
        throw std::runtime_error("failed to record compute command buffer!");
}

void PENNIS::recordBackpropCommandBuffer()
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(computeCommandBuffer, &beginInfo) != VK_SUCCESS)
        throw std::runtime_error("failed to begin recording compute command buffer!");

    vkCmdBindPipeline(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, backpropPipeline);

    bool isOutput = true;

    struct Push { uint32_t inSize, outSize, actType, isOutput, phase; } push;

    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--)
    {
        Layer& L = layers[i];

        vkCmdBindDescriptorSets(
            computeCommandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            backpropPipelineLayout,
            0, 1,
            &backpropDescriptorSets[i],
            0, nullptr);

        push = { L.inSize, L.outSize, L.actType, static_cast<uint32_t>(isOutput), 0 };

        vkCmdPushConstants(computeCommandBuffer, backpropPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

        uint32_t groups = (L.outSize - 1) / workgroupSize + 1;
        vkCmdDispatch(computeCommandBuffer, groups, 1, 1);

        {
            VkMemoryBarrier memBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
            memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            vkCmdPipelineBarrier(
                computeCommandBuffer,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                1, &memBarrier,
                0, nullptr,
                0, nullptr
            );
            
            push.phase = 1;

            vkCmdPushConstants(computeCommandBuffer, backpropPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

            uint32_t groups = (L.inSize - 1) / workgroupSize + 1;
            vkCmdDispatch(computeCommandBuffer, groups, 1, 1);
            
            if(i > 0)
            {
                VkMemoryBarrier memBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
                memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

                vkCmdPipelineBarrier(
                    computeCommandBuffer,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    1, &memBarrier,
                    0, nullptr,
                    0, nullptr
                );
                
                VkBuffer src = L.dInput.buffer,
                         dst = layers[i - 1].dOutput.buffer;
                
                if (src == VK_NULL_HANDLE || dst == VK_NULL_HANDLE) continue;

                VkBufferCopy copyRegion{};
                copyRegion.srcOffset = 0;
                copyRegion.dstOffset = 0;
                copyRegion.size = L.inSize * sizeof(float);

                vkCmdCopyBuffer(computeCommandBuffer, src, dst, 1, &copyRegion);

                VkBufferMemoryBarrier bufBarrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
                bufBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
                bufBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
                bufBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                bufBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                bufBarrier.buffer = dst;
                bufBarrier.offset = 0;
                bufBarrier.size = VK_WHOLE_SIZE;

                vkCmdPipelineBarrier(
                    computeCommandBuffer,
                    VK_PIPELINE_STAGE_TRANSFER_BIT,
                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    0, nullptr,
                    1, &bufBarrier,
                    0, nullptr
                );
            }
        }

        if(isOutput) isOutput = false;
    }

    if (vkEndCommandBuffer(computeCommandBuffer) != VK_SUCCESS)
        throw std::runtime_error("failed to record compute command buffer!");
}

void PENNIS::recordAdamCommandBuffer()
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (vkBeginCommandBuffer(computeCommandBuffer, &beginInfo) != VK_SUCCESS)
        throw std::runtime_error("failed to begin recording compute command buffer!");

    vkCmdBindPipeline(computeCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, adamPipeline);

    struct Push { AdamParams adamParams; uint32_t t, size, phase; } push;

    push.adamParams = adamParams;
    push.t          = adamTimestep;

    push.phase = 0;

    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--)
    {
        Layer& L = layers[i];

        vkCmdBindDescriptorSets(
            computeCommandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            adamPipelineLayout,
            0, 1,
            &adamDescriptorSets[i],
            0, nullptr);
        
        push.size = L.inSize * L.outSize;
        vkCmdPushConstants(computeCommandBuffer, adamPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

        uint32_t groups = (L.inSize * L.outSize - 1) / workgroupSize + 1;
        vkCmdDispatch(computeCommandBuffer, groups, 1, 1);
    }

    VkMemoryBarrier memBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER };
    memBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memBarrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(
    computeCommandBuffer,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    0,
    1, &memBarrier,
    0, nullptr,
    0, nullptr
    );

    push.phase = 1;

    for (int i = static_cast<int>(layers.size()) - 1; i >= 0; i--)
    {
        Layer& L = layers[i];

        vkCmdBindDescriptorSets(
            computeCommandBuffer,
            VK_PIPELINE_BIND_POINT_COMPUTE,
            adamPipelineLayout,
            0, 1,
            &adamDescriptorSets[i],
            0, nullptr);

        push.size = L.outSize;
        vkCmdPushConstants(computeCommandBuffer, adamPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

        uint32_t groups = (L.outSize - 1) / workgroupSize + 1;
        vkCmdDispatch(computeCommandBuffer, groups, 1, 1);
    }

    if (vkEndCommandBuffer(computeCommandBuffer) != VK_SUCCESS)
        throw std::runtime_error("failed to record compute command buffer!");
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