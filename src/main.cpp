// src/main.cpp

#include <iostream>
#include "model/zyraAI_model.h"
#include "personalization/personality_manager.h"

int main() {
    std::cout << "ZyraAI Starting..." << std::endl;
    zyraai::ZyraAIModel model;
    model.train();
    model.predict();

    zyraai::PersonalityManager personalityManager;
    personalityManager.addPersonality("normal", "normal_behavior_script");
    personalityManager.addPersonality("anime_waifu", "anime_waifu_behavior_script");

    personalityManager.setActivePersonality("normal");
    std::cout << personalityManager.getResponse("Hello!") << std::endl;

    personalityManager.setActivePersonality("anime_waifu");
    std::cout << personalityManager.getResponse("Hello Senpai!") << std::endl;

    return 0;
}
