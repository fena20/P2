#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست سریع Reward Function برای بررسی رفتار Agent

این اسکریپت reward function را در سناریوهای مختلف test می‌کند
بدون نیاز به training کامل.

استفاده:
    python test_reward_function.py
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict

# =========================================================================
# CONFIG
# =========================================================================

@dataclass
class Config:
    setpoint: float = 21.0
    deadband: float = 1.5
    comfort_min: float = 19.5
    comfort_max: float = 24.0
    
    Q_hvac_kw: float = 3.0
    
    peak_price: float = 0.30
    offpeak_price: float = 0.10
    
    # Weights - از balanced version
    w_comfort_violation: float = 50.0
    w_temp_deviation: float = 2.0
    w_unnecessary_on: float = 5.0
    w_cost: float = 1.0
    w_peak: float = 2.0
    w_switch: float = 0.1


# =========================================================================
# REWARD HANDLER
# =========================================================================

class BalancedRewardHandler:
    def __init__(self, config: Config):
        self.config = config
        self.setpoint = config.setpoint
        self.deadband = config.deadband
        self.comfort_min = config.comfort_min
        self.comfort_max = config.comfort_max
        
        self.lower_deadband = self.setpoint - self.deadband
        self.upper_deadband = self.setpoint + self.deadband

    def calculate(
        self,
        T_in: float,
        action: int,
        price_t: float,
        prev_action: int,
        is_peak: bool
    ) -> Tuple[float, Dict]:

        dt_hours = 1.0 / 60.0
        energy_kwh = (action * self.config.Q_hvac_kw) * dt_hours
        instant_cost = energy_kwh * price_t

        in_deadband = (self.lower_deadband <= T_in <= self.upper_deadband)
        in_comfort_band = (self.comfort_min <= T_in <= self.comfort_max)
        
        # Comfort penalty
        comfort_penalty = 0.0
        
        if not in_comfort_band:
            if T_in < self.comfort_min:
                violation = self.comfort_min - T_in
            else:
                violation = T_in - self.comfort_max
            comfort_penalty = self.config.w_comfort_violation * (violation ** 2)
            
        elif not in_deadband:
            if T_in < self.lower_deadband:
                deviation = self.lower_deadband - T_in
            else:
                deviation = T_in - self.upper_deadband
            comfort_penalty = self.config.w_temp_deviation * (deviation ** 2)
        
        # Unnecessary ON penalty
        unnecessary_on_penalty = 0.0
        if in_deadband and action == 1 and T_in >= self.setpoint:
            unnecessary_on_penalty = self.config.w_unnecessary_on
        
        # Cost
        cost_term = self.config.w_cost * instant_cost
        
        # Peak
        peak_penalty = 0.0
        if is_peak and action == 1 and in_deadband:
            peak_penalty = self.config.w_peak * energy_kwh
        
        # Switch
        switch_penalty = 0.0
        if action != prev_action:
            switch_penalty = self.config.w_switch
        
        baseline_reward = 1.0 if in_comfort_band else 0.0
        
        total_penalty = (
            comfort_penalty +
            unnecessary_on_penalty +
            cost_term +
            peak_penalty +
            switch_penalty
        )
        
        reward = baseline_reward - total_penalty

        return reward, {
            "comfort_penalty": comfort_penalty,
            "unnecessary_on": unnecessary_on_penalty,
            "cost_term": cost_term,
            "peak_penalty": peak_penalty,
            "switch_penalty": switch_penalty,
            "baseline_reward": baseline_reward,
            "total_penalty": total_penalty,
            "in_deadband": in_deadband,
            "in_comfort": in_comfort_band,
        }


# =========================================================================
# TEST SCENARIOS
# =========================================================================

def test_scenario(handler, scenario_name, T_in, action, price, is_peak, prev_action=0):
    """Test یک سناریو و نمایش نتایج"""
    
    reward, comp = handler.calculate(
        T_in=T_in,
        action=action,
        price_t=price,
        prev_action=prev_action,
        is_peak=is_peak
    )
    
    action_str = "ON" if action == 1 else "OFF"
    price_str = "peak" if is_peak else "off-peak"
    
    print(f"\n{'=' * 70}")
    print(f"Scenario: {scenario_name}")
    print(f"{'=' * 70}")
    print(f"Temperature: {T_in:.1f}°C")
    print(f"Action: {action_str}")
    print(f"Price: ${price:.2f}/kWh ({price_str})")
    print(f"\nStatus:")
    print(f"  In comfort band? {'✅ Yes' if comp['in_comfort'] else '❌ No'}")
    print(f"  In deadband? {'✅ Yes' if comp['in_deadband'] else '❌ No'}")
    print(f"\nReward components:")
    print(f"  Baseline reward:        {comp['baseline_reward']:+8.4f}")
    print(f"  Comfort penalty:        {comp['comfort_penalty']:+8.4f}")
    print(f"  Unnecessary ON penalty: {comp['unnecessary_on']:+8.4f}")
    print(f"  Cost term:              {comp['cost_term']:+8.4f}")
    print(f"  Peak penalty:           {comp['peak_penalty']:+8.4f}")
    print(f"  Switch penalty:         {comp['switch_penalty']:+8.4f}")
    print(f"  {'─' * 50}")
    print(f"  Total penalty:          {comp['total_penalty']:+8.4f}")
    print(f"  FINAL REWARD:           {reward:+8.4f}")
    
    return reward


def main():
    print("=" * 70)
    print("TEST: Balanced Reward Function")
    print("=" * 70)
    
    config = Config()
    handler = BalancedRewardHandler(config)
    
    print(f"\nConfiguration:")
    print(f"  Setpoint: {config.setpoint}°C")
    print(f"  Deadband: [{config.setpoint - config.deadband}, {config.setpoint + config.deadband}]°C")
    print(f"  Comfort band: [{config.comfort_min}, {config.comfort_max}]°C")
    print(f"\nWeights:")
    print(f"  w_comfort_violation = {config.w_comfort_violation}")
    print(f"  w_temp_deviation = {config.w_temp_deviation}")
    print(f"  w_unnecessary_on = {config.w_unnecessary_on}")
    print(f"  w_cost = {config.w_cost}")
    print(f"  w_peak = {config.w_peak}")
    
    # Test scenarios
    scenarios = []
    
    # Scenario 1: در deadband، زیر setpoint
    scenarios.append({
        "name": "در deadband، T < setpoint",
        "T_in": 20.0,
        "tests": [
            ("OFF", 0, False),
            ("ON", 1, False),
        ]
    })
    
    # Scenario 2: در deadband، بالای setpoint
    scenarios.append({
        "name": "در deadband، T >= setpoint",
        "T_in": 21.5,
        "tests": [
            ("OFF", 0, False),
            ("ON (unnecessary!)", 1, False),
        ]
    })
    
    # Scenario 3: زیر deadband
    scenarios.append({
        "name": "زیر deadband (نیاز به گرمایش)",
        "T_in": 19.0,
        "tests": [
            ("OFF (بد!)", 0, False),
            ("ON (خوب)", 1, False),
        ]
    })
    
    # Scenario 4: خارج comfort band
    scenarios.append({
        "name": "خارج comfort band (اورژانسی!)",
        "T_in": 18.0,
        "tests": [
            ("OFF (فاجعه!)", 0, False),
            ("ON (ضروری)", 1, False),
        ]
    })
    
    # Scenario 5: در deadband + peak hours
    scenarios.append({
        "name": "در deadband + Peak Hours",
        "T_in": 21.5,
        "tests": [
            ("OFF (خوب - peak shaving)", 0, True),
            ("ON (بد - unnecessary + peak)", 1, True),
        ]
    })
    
    # Run all scenarios
    for scenario in scenarios:
        print("\n" + "=" * 70)
        print(f"SCENARIO GROUP: {scenario['name']}")
        print("=" * 70)
        
        results = []
        for test_name, action, is_peak in scenario['tests']:
            price = config.peak_price if is_peak else config.offpeak_price
            reward = test_scenario(
                handler,
                f"{scenario['name']} - {test_name}",
                T_in=scenario['T_in'],
                action=action,
                price=price,
                is_peak=is_peak
            )
            results.append((test_name, reward))
        
        # Compare
        print(f"\n{'─' * 70}")
        print("COMPARISON:")
        for test_name, reward in results:
            print(f"  {test_name:30s} → reward = {reward:+8.4f}")
        
        if len(results) == 2:
            if results[1][1] > results[0][1]:
                better = results[1][0]
            else:
                better = results[0][0]
            print(f"\n  ✅ Agent should prefer: {better}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY & EXPECTED BEHAVIOR")
    print("=" * 70)
    print("""
Based on these tests, the agent should learn to:

1. ✅ Stay OFF when T is in deadband AND above setpoint
   → Saves energy without comfort penalty

2. ✅ Turn ON when T drops below deadband
   → Prevents discomfort

3. ✅ Turn ON immediately when T goes below comfort band
   → Emergency response (large penalty if OFF)

4. ✅ Prefer OFF during peak hours (if in deadband)
   → Peak shaving

5. ✅ NOT stay ON all the time
   → Unnecessary ON penalty prevents Always-ON behavior

This should result in:
  - Cycling behavior (ON/OFF/ON/OFF...)
  - Cost reduction vs baseline
  - Good comfort maintenance
  - Peak shaving during 16-21 hours

If training results don't match this, adjust weights!
""")


if __name__ == "__main__":
    main()
