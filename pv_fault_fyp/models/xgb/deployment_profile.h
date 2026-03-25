#pragma once

// Auto-generated deployment profile
// Final hardware-oriented thresholds and feature metadata

#define PV_WINDOW_SIZE 5
#define PV_ENTER_TH 0.450000f
#define PV_EXIT_TH 0.250000f
#define PV_K_ON 3
#define PV_K_OFF 3

#define PV_SHADE_TH 0.550000f
#define PV_SHADE_FAULT_PROB_MIN 0.550000f
#define PV_SHADE_POWER_REL_MAX 0.800000f
#define PV_SHADE_CURRENT_REL_MAX 0.850000f
#define PV_SHADE_LOW_POWER_MIN 0.000000f

#define PV_OC_CURRENT_FRAC_TH 0.800000f
#define PV_OC_POWER_FRAC_TH 0.800000f

#define PV_FEATURE_COUNT 25

static const char* PV_FEATURE_NAMES[] = {
    "voltage_V_std",
    "voltage_V_slope",
    "voltage_V_range",
    "voltage_V_delta",
    "voltage_V_rel_mean",
    "voltage_V_rel_last",
    "voltage_V_pct_change_from_ref",
    "current_A_std",
    "current_A_slope",
    "current_A_range",
    "current_A_delta",
    "current_A_rel_mean",
    "current_A_rel_last",
    "current_A_pct_change_from_ref",
    "power_W_std",
    "power_W_slope",
    "power_W_range",
    "power_W_delta",
    "power_W_rel_mean",
    "power_W_rel_last",
    "power_W_pct_change_from_ref",
    "mean_power_over_voltage",
    "mean_current_over_voltage",
    "low_current_fraction",
    "low_power_fraction",
};

static const float PV_STAGE1_FILL_MEDIANS[] = {
    0.086162637f,
    -0.005000000f,
    0.240000000f,
    -0.020000000f,
    1.020138889f,
    1.020138889f,
    0.020138889f,
    0.035217530f,
    -0.000000000f,
    0.093800000f,
    -0.000100000f,
    0.969344124f,
    0.990216210f,
    -0.030655876f,
    0.525722360f,
    -0.000000000f,
    1.420000000f,
    0.000000000f,
    0.995932203f,
    1.003389831f,
    -0.004067797f,
    0.403016049f,
    0.027430256f,
    0.000000000f,
    0.000000000f,
};

static const float PV_STAGE2_FILL_MEDIANS[] = {
    0.077562878f,
    -0.006000000f,
    0.220000000f,
    -0.020000000f,
    1.001250000f,
    1.002083333f,
    0.001250000f,
    0.021436642f,
    -0.000000000f,
    0.059900000f,
    0.000000000f,
    0.894745742f,
    0.908805411f,
    -0.105254258f,
    0.290006896f,
    -0.000000000f,
    0.810000000f,
    0.000000000f,
    0.859322034f,
    0.862711864f,
    -0.140677966f,
    0.371893357f,
    0.026948370f,
    0.000000000f,
    0.000000000f,
};
