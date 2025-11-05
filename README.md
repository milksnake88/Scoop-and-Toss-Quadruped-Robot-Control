# 🐾 Scoop-and-Toss: Dynamic Object Collection for Quadrupedal Systems

**Quadruped robots** have made significant progress in locomotion, extending their abilities from controlled settings to real-world environments.  
This project proposes a **hierarchical reinforcement learning framework** that enables a quadruped robot to collect multiple objects using only its legs—without any additional actuators.

---

## 🧠 Overview

In this work, the robot is equipped with a simple **scoop-like add-on** on one leg and a **collection tray** mounted on its back.  
The framework allows the robot to **scoop objects on the ground and toss them into the tray** by leveraging leg agility.  

We design a **hierarchical policy structure** consisting of:
- **Scoop-and-Toss Policy (π<sub>scoop-toss</sub>)** — learns how to scoop and toss an object into the tray.  
- **Approach Policy (π<sub>approach</sub>)** — learns to approach the target object’s position.  
- **Meta Policy (π<sub>meta</sub>)** — dynamically selects which expert policy to use based on the current state, enabling **coordinated multi-object collection**.

---

## 🏋️‍♂️ Method

- **Algorithm:** Proximal Policy Optimization
- **Framework:** Hierarchical Reinforcement Learning  
- **Curriculum Learning:** The scoop-and-toss policy is trained with gradually increasing object placement randomness for robust behavior.  
- **State Transition-based Initialization (STI):** Used to fine-tune expert policies for smoother transitions between behaviors.  

---

## ⚙️ Simulation Setup

- **Simulator:** [Isaac Gym](https://developer.nvidia.com/isaac-gym)  
- **Environment Count:** 4,096 parallel environments (GPU accelerated)  
- **Hardware:**  
  - NVIDIA RTX 3090 (π<sub>scoop-toss</sub>)  
  - NVIDIA RTX 4070 (π<sub>approach</sub> & π<sub>meta</sub>)  
- **Training Steps:**  
  - π<sub>scoop-toss</sub>: 28k steps (~85 hours)  
  - π<sub>approach</sub>: 7k steps (~5 hours)  
  - π<sub>meta</sub>: 6k steps (~12 hours)  
- **Object:** Cube (4 cm side, 96 g weight)  

---

## 🧩 Results

The trained policies enable the quadruped robot to autonomously:
- Approach scattered objects on the ground  
- Scoop them up using its leg-mounted add-on  
- Toss them into the tray on its back  
- Repeat this sequence for **multiple objects** with smooth transitions  

![VideoProject5-ezgif com-crop](https://github.com/user-attachments/assets/2cb24c72-f2fa-4a3a-9c38-f6f4fec42aad)

🎥 **Full Video**  

For the complete results — including expert policy experiments, various-object experiments, and joy-stick control —  
please refer to the full video below:

🔗 [Watch on YouTube](https://youtu.be/uiymKDvBhqo?si=0NBjBdXMmXQbp0HV)

---

## 🧠 Keywords

`Quadruped Robot` `Hierarchical Reinforcement Learning` `Isaac Gym`  
`PPO` `Curriculum Learning` `Object Manipulation`

---

