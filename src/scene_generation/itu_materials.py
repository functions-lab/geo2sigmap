"""
ITU Material Properties Data

This module provides predefined material properties based on
ITU-R Recommendation P.2040-2: "[Effects of building materials and structures
on radiowave propagation above about 100 MHz](https://www.itu.int/dms_pubrec/itu-r/rec/p/R-REC-P.2040-3-202308-I!!PDF-E.pdf)".

TODO: Verify the exact material names using by Mitsuba and Sionna. Especially how they named the low/high frequency glass and ceiling board materials.
TODO: Check what is the exact colors using by Mitsuba/Sionna/Blender/Blosm for the materials.
"""

ITU_MATERIALS = {
    "mat-itu_vacuum": {
        "name": "Vacuum (\u2248Air)",
        "lower_freq_limit": 0.001e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color": (0.1216, 0.4667, 0.7059),
    },
    "mat-itu_concrete": {
        "name": "Concrete",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color": (0.539479, 0.539479, 0.539480),
    },
    "mat-itu_brick": {
        "name": "Brick",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 40e9,
        "mitsuba_color":(1.0000, 0.4980, 0.0549),
    },
    "mat-itu_plasterboard": {
        "name": "Plasterboard",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color":(1.0000, 0.7333, 0.4706),
    },
    "mat-itu_wood": {
        "name": "Wood",
        "lower_freq_limit": 0.001e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color": (0.043, 0.58, 0.184)
    },
    "mat-itu_glass_low_freq": {
        "name": "Glass(Low Frequency)",
        "lower_freq_limit": 0.1e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color":(0.5961, 0.8745, 0.5412),
    },
    "mat-itu_glass_high_freq": {
        "name": "Glass(High Frequency)",
        "lower_freq_limit": 220e9,
        "upper_freq_limit": 450e9,
        "mitsuba_color":(0.8392, 0.1529, 0.1569),
    },
    "mat-itu_ceiling_board_low_freq": {
        "name": "Ceiling Board(Low Frequency)",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color":(1.0000, 0.5961, 0.5882),
    },
    "mat-itu_ceiling_board_high_freq": {
        "name": "Ceiling Board(High Frequency)",
        "lower_freq_limit": 220e9,
        "upper_freq_limit": 450e9,
        "mitsuba_color": (0.5804, 0.4039, 0.7412),
    },
    "mat-itu_chipboard": {
        "name": "Chipboard",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color":(0.7725, 0.6902, 0.8353),
    },
    "mat-itu_plywood": {
        "name": "Plywood",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 40e9,
        "mitsuba_color":(0.5490, 0.3373, 0.2941),
    },
    "mat-itu_marble": {
        "name": "Marble",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 60e9,
        "mitsuba_color": (0.701101, 0.644479, 0.485150),
    },
    "mat-itu_floorboard": {
        "name": "Floorboard",
        "lower_freq_limit": 50e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color":(0.8902, 0.4667, 0.7608),
    },
    "mat-itu_metal": {
        "name": "Metal",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 100e9,
        "mitsuba_color": (0.219526, 0.219526, 0.254152)
    },
    "mat-itu_very_dry_ground": {
        "name": "Very Dry Ground",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 10e9,
        "mitsuba_color": (0.4980, 0.4980, 0.4980),
    },
    "mat-itu_medium_dry_ground": {
        "name": "Medium Dry Ground",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 10e9,
        "mitsuba_color":(0.7804, 0.7804, 0.7804),
    },
    "mat-itu_wet_ground": {
        "name": "Wet Ground",
        "lower_freq_limit": 1e9,
        "upper_freq_limit": 10e9,
        "mitsuba_color": (0.91, 0.569, 0.055)
    },
}
