#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
تست سریع Reward Function - نسخه ساده بدون dependency

این اسکریپت reward function را test می‌کند تا ببینیم
آیا رفتار مورد انتظار را دارد یا نه.
"""

# =========================================================================
# CONFIG
# =========================================================================

class Config:
    setpoint = 21.0
    deadband = 1.5
    comfort_min = 19.5
    comfort_max = 24.0
    
    Q_hvac_kw = 3.0
    
    peak_price = 0.30
    offpeak_price = 0.10
    
    # Weights
    w_comfort_violation = 50.0
    w_temp_deviation = 2.0
    w_unnecessary_on = 5.0
    w_cost = 1.0
    w_peak = 2.0
    w_switch = 0.1


# =========================================================================
# REWARD HANDLER
# =========================================================================

class BalancedRewardHandler:
    def __init__(self, config):
        self.config = config
        self.setpoint = config.setpoint
        self.deadband = config.deadband
        self.comfort_min = config.comfort_min
        self.comfort_max = config.comfort_max
        
        self.lower_deadband = self.setpoint - self.deadband
        self.upper_deadband = self.setpoint + self.deadband

    def calculate(self, T_in, action, price_t, prev_action, is_peak):
        dt_hours = 1.0 / 60.0
        energy_kwh = (action * self.config.Q_hvac_kw) * dt_hours
        instant_cost = energy_kwh * price_t

        in_deadband = (self.lower_deadband <= T_in <= self.upper_deadband)
        in_comfort_band = (self.comfort_min <= T_in <= self.comfort_max)
        
        # Comfort penalty با action-aware logic
        comfort_penalty = 0.0
        
        if not in_comfort_band:
            if T_in < self.comfort_min:
                violation = self.comfort_min - T_in
                # اگر سرد است و OFF است → جریمه cubic (بیشتر!)
                if action == 0:
                    comfort_penalty = self.config.w_comfort_violation * (violation ** 3)
                else:
                    comfort_penalty = self.config.w_comfort_violation * (violation ** 2)
            else:
                violation = T_in - self.comfort_max
                comfort_penalty = self.config.w_comfort_violation * (violation ** 2)
            
        elif not in_deadband:
            if T_in < self.lower_deadband:
                deviation = self.lower_deadband - T_in
                # اگر سرد است و OFF است → جریمه cubic
                if action == 0:
                    comfort_penalty = self.config.w_temp_deviation * (deviation ** 3)
                else:
                    comfort_penalty = self.config.w_temp_deviation * (deviation ** 2)
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
# TEST
# =========================================================================

def test_scenario(handler, name, T_in, action, price, is_peak, prev_action=0):
    """Test یک سناریو"""
    
    reward, comp = handler.calculate(T_in, action, price, prev_action, is_peak)
    
    action_str = "ON" if action == 1 else "OFF"
    price_str = "peak" if is_peak else "off-peak"
    
    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")
    print(f"T = {T_in:.1f}°C, Action = {action_str}, Price = {price_str}")
    print(f"\nStatus:")
    print(f"  In comfort? {'✅' if comp['in_comfort'] else '❌'}")
    print(f"  In deadband? {'✅' if comp['in_deadband'] else '❌'}")
    print(f"\nReward breakdown:")
    print(f"  Baseline:      {comp['baseline_reward']:+7.3f}")
    print(f"  Comfort:       {comp['comfort_penalty']:+7.3f}")
    print(f"  Unnecessary:   {comp['unnecessary_on']:+7.3f}")
    print(f"  Cost:          {comp['cost_term']:+7.3f}")
    print(f"  Peak:          {comp['peak_penalty']:+7.3f}")
    print(f"  Switch:        {comp['switch_penalty']:+7.3f}")
    print(f"  {'─' * 40}")
    print(f"  FINAL REWARD:  {reward:+7.3f}")
    
    return reward


def main():
    print("=" * 60)
    print("TEST: Balanced Reward Function")
    print("=" * 60)
    
    config = Config()
    handler = BalancedRewardHandler(config)
    
    print(f"\nSetup:")
    print(f"  Setpoint: {config.setpoint}°C")
    print(f"  Deadband: [{config.setpoint - config.deadband}, "
          f"{config.setpoint + config.deadband}]°C")
    print(f"  Comfort: [{config.comfort_min}, {config.comfort_max}]°C")
    
    # Test 1: در deadband، زیر setpoint
    print("\n" + "=" * 60)
    print("TEST 1: T در deadband، زیر setpoint (20°C)")
    print("=" * 60)
    
    r1 = test_scenario(handler, "Scenario 1A: OFF", 20.0, 0, 0.10, False)
    r2 = test_scenario(handler, "Scenario 1B: ON", 20.0, 1, 0.10, False)
    
    print(f"\n→ هر دو OK! OFF کمی بهتر (cost کمتر)")
    print(f"   OFF: {r1:+.3f}  vs  ON: {r2:+.3f}")
    if r1 > r2:
        print("   ✅ Agent ترجیح می‌دهد OFF باشد (energy saving)")
    
    # Test 2: در deadband، بالای setpoint
    print("\n" + "=" * 60)
    print("TEST 2: T در deadband، بالای setpoint (21.5°C)")
    print("=" * 60)
    
    r3 = test_scenario(handler, "Scenario 2A: OFF", 21.5, 0, 0.10, False)
    r4 = test_scenario(handler, "Scenario 2B: ON (unnecessary!)", 21.5, 1, 0.10, False)
    
    print(f"\n→ OFF خیلی بهتر! ON غیرضروری است")
    print(f"   OFF: {r3:+.3f}  vs  ON: {r4:+.3f}")
    if r3 > r4:
        print("   ✅ Agent یاد می‌گیرد در deadband بالا OFF باشد")
    else:
        print("   ❌ مشکل! w_unnecessary_on را افزایش دهید")
    
    # Test 3: زیر deadband (اما داخل comfort)
    print("\n" + "=" * 60)
    print("TEST 3: T زیر deadband اما داخل comfort (20°C)")
    print("=" * 60)
    
    r5 = test_scenario(handler, "Scenario 3A: OFF", 20.0, 0, 0.10, False)
    r6 = test_scenario(handler, "Scenario 3B: ON", 20.0, 1, 0.10, False)
    
    print(f"\n→ هر دو OK (در deadband)")
    print(f"   OFF: {r5:+.3f}  vs  ON: {r6:+.3f}")
    if r5 >= r6 - 0.2:
        print("   ✅ Agent می‌تواند انتخاب کند (OFF کمی بهتر)")
    else:
        print("   ⚠️  ON penalty زیاد است")
    
    # Test 4: خارج comfort
    print("\n" + "=" * 60)
    print("TEST 4: T خارج comfort (18°C) - اورژانسی!")
    print("=" * 60)
    
    r7 = test_scenario(handler, "Scenario 4A: OFF (فاجعه!)", 18.0, 0, 0.10, False)
    r8 = test_scenario(handler, "Scenario 4B: ON (ضروری)", 18.0, 1, 0.10, False)
    
    print(f"\n→ ON خیلی خیلی بهتر! OFF فاجعه است")
    print(f"   OFF: {r7:+.3f}  vs  ON: {r8:+.3f}")
    if r8 > r7:
        print("   ✅ Agent یاد می‌گیرد خارج comfort فوراً ON کند")
    else:
        print("   ❌ مشکل! w_comfort_violation را افزایش دهید")
    
    # Test 5: Peak hours
    print("\n" + "=" * 60)
    print("TEST 5: T در deadband بالا (21.5°C) + Peak Hours")
    print("=" * 60)
    
    r9 = test_scenario(handler, "Scenario 5A: OFF (peak shaving)", 21.5, 0, 0.30, True)
    r10 = test_scenario(handler, "Scenario 5B: ON (costly!)", 21.5, 1, 0.30, True)
    
    print(f"\n→ OFF خیلی بهتر! (unnecessary + peak + cost)")
    print(f"   OFF: {r9:+.3f}  vs  ON: {r10:+.3f}")
    if r9 > r10:
        print("   ✅ Agent یاد می‌گیرد در peak hours OFF باشد")
    else:
        print("   ❌ مشکل! w_peak را افزایش دهید")
    
    # Summary
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 5
    
    checks = [
        (r1 > r2 - 0.1, "Test 1: OFF/ON در deadband پایین"),
        (r3 > r4, "Test 2: OFF > ON در deadband بالا (key!)"),
        (r5 >= r6 - 0.2, "Test 3: OFF/ON زیر setpoint OK"),
        (r8 > r7, "Test 4: ON >> OFF خارج comfort"),
        (r9 > r10, "Test 5: OFF > ON در peak hours"),
    ]
    
    for passed, name in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {name}")
        if passed:
            tests_passed += 1
    
    print(f"\n{'─' * 60}")
    print(f"Score: {tests_passed}/{tests_total}")
    
    if tests_passed == tests_total:
        print("\n✅✅✅ EXCELLENT! Reward function خوب طراحی شده")
        print("     Agent باید رفتار مطلوب را یاد بگیرد:")
        print("       - Cycling (ON/OFF/ON/OFF...)")
        print("       - OFF در deadband بالا")
        print("       - ON زیر deadband")
        print("       - Peak shaving")
    elif tests_passed >= 3:
        print("\n⚠️  PARTIAL: بعضی تست‌ها OK، بعضی نه")
        print("    نگاه کنید به FAIL ها و weights را adjust کنید")
    else:
        print("\n❌ FAIL: Reward function نیاز به تنظیم دارد")
        print("   Weights را بر اساس FAIL ها تغییر دهید")


if __name__ == "__main__":
    main()
