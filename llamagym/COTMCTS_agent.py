from collections import Counter
from llamagym import Agent
import gymnasium as gym
import re
import torch


class COTMCTSBlackjackAgent(Agent):
    def __init__(self, model, tokenizer, judge_model, judge_tokenizer, device, num_cot_samples=5):
        super().__init__(model, tokenizer, device)
        self.judge_model = judge_model
        self.judge_tokenizer = judge_tokenizer
        self.num_cot_samples = num_cot_samples
        
    def get_system_prompt(self) -> str:
        """ COT WITH LLM [TODO] NEED BETTER PROMPT LATER? WE HAVE TO TUNE THIS PROMPT! """
        return (
            "You are an expert blackjack player. Think step by step before deciding whether to hit or stick. "
            "Consider your current total, the dealer’s visible card, and whether you have an ace.\n"
            "Write your reasoning first, then decide the action using 'Action: 0' or 'Action: 1'."
        )
    
    def generate_cot_answers(self, messages: List[Dict[str, str]]) -> List[str]:
        """ Generate multiple COT answers using LLM """
        cot_answers = []
        for _ in range(self.num_cot_samples):
            response = self.llm(messages)
            cot_answers.append(response)
        return cot_answers
    
    def judge_answers(self, messages: List[Dict[str, str]], cot_answers: List[str]) -> str:
        """ Judge which COT answer is the best using LLM """
        judge_prompt = "You are evaluating multiple AI responses for the best blackjack strategy."
        judge_messages = [{"role": "system", "content": judge_prompt}]
        
        judge_text = f"User's situation: {messages[-1]['content']}\n"
        for i, answer in enumerate(cot_answers):
            judge_text += f"\n### Answer {i+1}:\n{answer}\n"
        judge_text += "\nChoose the best response (return number only)."

        judge_messages.append({"role": "user", "content": judge_text})
        best_idx = int(self.llm(judge_messages).strip()) - 1 
        return cot_answers[best_idx]

    def act(self, observation):
        """ COT + MCTS"""
        message = self.format_observation(observation)
        self.current_episode_messages.append({"role": "user", "content": message})

        cot_answers = self.generate_cot_answers(self.current_episode_messages)
        best_response = self.judge_answers(self.current_episode_messages, cot_answers)
        action = self.extract_action(best_response)
        self.current_episode_messages.append({"role": "assistant", "content": best_response})
        return action