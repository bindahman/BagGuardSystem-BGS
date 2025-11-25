Bag-Guard-System-BGS-
“Secure the bag, secure the space.”

An AI-powered airport security system that detects unattended luggage and identifies suspicious behaviour, helping prevent theft and improving real-time airport surveillance accuracy.

Project Overview

The Bag Guard System (BGS) is designed to enhance airport safety by automatically identifying unattended luggage and analysing human–bag interactions using intelligent computer vision.
The system monitors luggage and nearby individuals to reduce the risk of:

Theft

Suspicious behaviour

Unattended or forgotten bags

Slow or delayed security response

False alarms in CCTV monitoring

The project is simulated using Python and GDPR-safe video data.

Key Features

AI Luggage Detection: Identifies suitcases, backpacks, and bags in real time.

Owner Association Tracking: Links bags to nearby individuals using proximity and interaction cues.

Unattended Bag Alerts: Flags luggage left behind beyond a defined time and distance.

Suspicious Behaviour Analysis: Detects unauthorised lifting, loitering, or potential theft actions.

Real-Time Monitoring: Displays alerts, tracking IDs, and interactions on a security dashboard.

Aims

To design an AI-driven system that detects unattended luggage, verifies owner interactions, and identifies suspicious behaviour, improving airport security through fast, reliable alerts.

Objectives

Detect luggage and people using YOLO-based models.

Track movement patterns using object tracking.

Confirm ownership through proximity and interactions.

Identify unattended luggage using rule-based logic.

Recognise suspicious behaviour that may indicate theft.

Display alerts and visual cues for security staff.

Use of AI

AI powers luggage and person detection, tracking, ownership analysis, and behaviour recognition. The system avoids biometric data and focuses on explainable, object-based analysis to support real-time airport security.

Legal, Ethical & Professional

GDPR-compliant: no facial recognition or sensitive data used.

Transparent, fair AI logic with minimal bias.

Supports human operators rather than replacing them.

Follows BCS and Engineering Council ethical guidelines.

Ensures responsible use of public surveillance data.

Technologies Used

Python

OpenCV

YOLOv8 / YOLOv11

DeepSORT / ByteTrack

MediaPipe or OpenPose (optional)

Pandas, NumPy

Jupyter Notebook

References

[1] Intel Corporation, Unattended Baggage Detection Using Deep Neural Networks, 2020.
[2] A. Al-Zahrani, “A System for Real-Time Detection of Abandoned Luggage,” Sensors, 2025.
[3] M. S. Nixon, “Robust Abandoned Object Detection for Surveillance,” University of Reading, 2012.
[4] BBC News, “Amazon’s ‘Just Walk Out’ AI Tracks Shopper Behaviour,” 2021.
