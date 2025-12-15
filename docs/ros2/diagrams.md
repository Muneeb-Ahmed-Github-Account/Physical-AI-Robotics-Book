---
title: ROS 2 Diagrams
sidebar_position: 7
description: Visual representations of ROS 2 concepts and architecture
---

# ROS 2 Diagrams

## Learning Objectives

After reviewing these diagrams, students will be able to:
- Visualize the layered architecture of ROS 2 and its components
- Understand the different communication patterns and their relationships
- Recognize Quality of Service policy types and their applications
- Identify the states and transitions in lifecycle nodes
- Comprehend the distributed nature of ROS 2 systems

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    ROS 2 Architecture                       │
├─────────────────────────────────────────────────────────────┤
│  Application Layer                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   Node A        │  │   Node B        │  │   Node C    │ │
│  │  (Publisher)    │  │  (Subscriber)   │  │ (Service)   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Client Library Layer (rclcpp, rclpy, etc.)                │
├─────────────────────────────────────────────────────────────┤
│  Middleware Layer (DDS Implementation)                     │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  RMW (ROS Middleware Wrapper)                           ││
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐ ││
│  │  │  Fast DDS     │ │  Cyclone DDS  │ │  RTI Connext │ ││
│  │  │  (Default)    │ │               │ │   DDS        │ ││
│  │  └───────────────┘ └───────────────┘ └───────────────┘ ││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  Operating System Layer (Linux, Windows, macOS, etc.)      │
└─────────────────────────────────────────────────────────────┘
```

## Communication Patterns

### Publisher-Subscriber Pattern
```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   Node A    │ Publish │   Topic     │ Publish │   Node B    │
│(Publisher)  │ ──────▶ │    /data    │ ──────▶ │(Subscriber) │
│             │         │             │         │             │
│  Publishes  │ ◀────── │ Subscription│ ◀────── │ Subscribes  │
│  messages   │         │   (QoS)     │         │   to data   │
└─────────────┘         └─────────────┘         └─────────────┘
```

### Service-Client Pattern
```
┌─────────────┐                    ┌─────────────┐
│   Node A    │      Request       │   Node B    │
│ (Client)    │ ─────────────────▶ │ (Service)   │
│             │                    │             │
│ Sends req-  │ ◀──────────────── │ Receives    │
│ uests      │      Response      │  requests   │
└─────────────┘                    └─────────────┘
```

### Action Pattern
```
┌─────────────┐                    ┌─────────────┐
│   Node A    │      Goal          │   Node B    │
│ (Client)    │ ─────────────────▶ │ (Server)    │
│             │                    │             │
│ Sends goals │ ◀──────────────── │ Receives    │
│ and gets    │   Feedback/Result  │  goals,     │
│ feedback &  │                    │  sends      │
│ result      │                    │  feedback   │
└─────────────┘                    └─────────────┘
```

## Quality of Service (QoS) Policies

```
┌─────────────────────────────────────────────────────────────┐
│                    QoS Policy Types                         │
├─────────────────────────────────────────────────────────────┤
│  Reliability Policy                                         │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │   RELIABLE      │  │  BEST_EFFORT    │                  │
│  │  (Guaranteed    │  │  (Try to send,  │                  │
│  │   delivery)     │  │   no guarantee) │                  │
│  └─────────────────┘  └─────────────────┘                  │
├─────────────────────────────────────────────────────────────┤
│  Durability Policy                                          │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │ TRANSIENT_LOCAL │  │    VOLATILE     │                  │
│  │(Keep historical │  │(No historical   │                  │
│  │ data for late   │  │  data)          │                  │
│  │ joiners)        │  │                 │                  │
│  └─────────────────┘  └─────────────────┘                  │
├─────────────────────────────────────────────────────────────┤
│  History Policy                                             │
│  ┌─────────────────┐  ┌─────────────────┐                  │
│  │  KEEP_LAST      │  │   KEEP_ALL      │                  │
│  │(Keep N messages)│  │(Keep all msg)   │                  │
│  └─────────────────┘  └─────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

## Lifecycle Node States

```
      ┌─────────────┐
      │ UNCONFIGURED│
      └──────┬──────┘
             │
      ┌──────▼──────┐
      │  CONFIGURE  │
      └──────┬──────┘
             │
      ┌──────▼──────┐
      │ INACTIVE    │
      └──────┬──────┘
             │
      ┌──────▼──────┐    ┌─────────────┐
      │ ACTIVATE    │───▶│ FINALIZED   │
      └──────┬──────┘    │(Shutdown)   │
             │           └─────────────┘
      ┌──────▼──────┐
      │   ACTIVE    │
      └──────┬──────┘
             │
      ┌──────▼──────┐
      │ DEACTIVATE  │
      └──────┬──────┘
             │
      ┌──────▼──────┐
      │  CLEANUP    │
      └─────────────┘
```

## Parameter Server Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Parameter System                         │
├─────────────────────────────────────────────────────────────┤
│  Parameter Server Node                                      │
│  ┌─────────────────────────────────────────────────────────┐│
│  │  Parameter Database                                     ││
│  │  ┌─────────────────┬─────────────────┬─────────────────┐││
│  │  │ Node Parameters │ System Params   │ User Config     │││
│  │  │                 │                 │                 │││
│  │  │ • control_rate  │ • log_level     │ • custom_val    │││
│  │  │ • safety_margin │ • use_sim_time  │ • threshold     │││
│  │  └─────────────────┴─────────────────┴─────────────────┘││
│  └─────────────────────────────────────────────────────────┘│
├─────────────────────────────────────────────────────────────┤
│  Node A                    Node B                   Node C  │
│  ┌─────────────────┐      ┌─────────────────┐      ┌───────┐│
│  │ Declares params │      │ Declares params │      │Gets/  ││
│  │ Gets/sets params│      │ Gets/sets params│      │sets   ││
│  │ with validation │      │ with validation │      │params ││
│  └─────────────────┘      └─────────────────┘      └───────┘│
└─────────────────────────────────────────────────────────────┘
```

## Distributed System Architecture

```
Network: 192.168.1.x
         ┌─────────────────┐    ┌─────────────────┐
Device A │   Humanoid      │    │   Workstation   │ Device B
ROS Dom  │   Robot         │◄──►│   (Simulation)  │ ROS Dom
ID: 42   │                 │    │                 │ ID: 42
         │ • Perception    │    │ • RViz          │
         │ • Control       │    │ • Gazebo        │
         │ • Navigation    │    │ • Dev Tools     │
         │ • Planning      │    │ • Data Analysis │
         └─────────────────┘    └─────────────────┘

         ┌─────────────────┐    ┌─────────────────┐
Device C │   External      │    │   Monitoring    │ Device D
ROS Dom  │   Sensors       │◄──►│   System        │ ROS Dom
ID: 42   │                 │    │                 │ ID: 42
         │ • Cameras       │    │ • Logging       │
         │ • LIDAR         │    │ • Diagnostics   │
         │ • IMU           │    │ • Visualization │
         │ • Force Sensors │    │ • Alerting      │
         └─────────────────┘    └─────────────────┘
```

## Cross-References

For related diagrams and visualizations, see:
- [Digital Twin Simulation Diagrams](../digital-twin/diagrams.md) for simulation architecture diagrams
- [NVIDIA Isaac Architecture](../nvidia-isaac/core-concepts.md) for platform-specific diagrams
- [Vision-Language-Action Architecture](../vla-systems/architecture.md) for AI system diagrams
- [Hardware Architecture](../hardware-guide/integration-diagrams-exercises.md) for hardware diagrams