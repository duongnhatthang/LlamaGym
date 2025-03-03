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
            "You are an expert blackjack player. Every turn, you'll see your current sum, the dealer's showing card value, "
            "and whether you have a usable ace. Your goal is to win by exceeding the dealer's hand but not exceeding 21.\n\n"
            "Decide whether to stay with your current sum by writing 'Action: 0' or accept another card by writing 'Action: 1'.\n\n"
            "Think step by step before making a decision:\n"
            "- Check your current total: If it's already close to 21, consider staying.\n"
            "- Evaluate the dealer's showing card: If the dealer has a weak card (2-6), they are more likely to bust.\n"
            "- Consider your ace status: If you have a usable ace, you have flexibility since it can be 1 or 11.\n"
            "- Compare risk vs reward: If hitting increases your bust risk too much, stay. Otherwise, take another card.\n\n"
            "Always maximize your winning probability based on Blackjack strategy."
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
        judge_prompt = judge_prompt = (
            "You are evaluating multiple AI responses for the best blackjack strategy. Every turn, the AI receives its current sum, "
            "the dealer's showing card value, and whether it has a usable ace. The AI must decide whether to hit ('Action: 1') "
            "or stay ('Action: 0') to maximize its chances of winning without exceeding 21.\n\n"
            "Use the following game rules to evaluate the AI’s decision-making:\n"
            "- The dealer must hit until reaching at least 17.\n"
            "- If the AI exceeds 21, it busts and automatically loses.\n"
            "- If the AI has a usable ace, it can choose between 1 and 11 for flexibility.\n"
            "- A higher sum is better unless it increases the risk of busting.\n\n"
            "Thinking step by step:\n"
            "1. What is the AI's current total? If it is already 21, staying is the best option.\n"
            "2. What is the dealer's showing card? If it‘s high (7-Ace), the AI may need a stronger total.\n"
            "3. What is the AI's bust risk? If hitting would likely cause a bust, staying is better.\n"
            "4. Was the AI's decision optimal based on these factors? Evaluate whether the AI followed good strategy.\n\n"
            "Provide an objective assessment based on these criteria."
        )
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