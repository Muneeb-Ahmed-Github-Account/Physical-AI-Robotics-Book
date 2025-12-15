# Humanoid Robotics & Physical AI Course Book

This repository contains the source content for the Humanoid Robotics & Physical AI Course Book, built with Docusaurus.

## Installation

```bash
npm install
```

## Local Development

```bash
npm start
```

This command starts a local development server and opens up a browser window. Most changes are reflected live without having to restart the server.

## Build

```bash
npm run build
```

This command generates static content into the `build` directory and can be served using any static contents hosting service.

## Validation Scripts

The following validation scripts are available to ensure content quality:

### Citation Validation
```bash
node scripts/citation-validator.js
```
Checks that all technical claims have proper numbered citations and that chapters follow the required structure.

### Cross-Reference Validation
```bash
node scripts/cross-ref-validator.js
```
Validates that cross-references between chapters are valid and functional.

## Directory Structure

- `/docs`: Contains all the course book content organized by modules
- `/src`: Contains custom React components and CSS
- `/scripts`: Contains validation and utility scripts
- `/specs`: Contains specification and planning documents for the book

## Course Modules

The book is organized into the following modules:

- **ROS 2 Fundamentals**: Core concepts and architecture of the Robot Operating System
- **Digital Twin Simulation**: Gazebo and Unity simulation environments for robot development
- **NVIDIA Isaac**: Advanced robotics platform for AI-powered robots
- **Vision-Language-Action Systems**: Integration of perception, reasoning, and action
- **Capstone Humanoid Project**: Complete implementation of a humanoid robot system
- **Hardware Guide**: Practical guidance on workstations, Jetson kits, sensors, and robot platforms

Each module follows the structure: overview → theory → implementation → examples → applications.

## Contributing

When adding new content, please follow these guidelines:

1. Use the chapter template (`docs/chapter-template.md`) as a starting point
2. Ensure all technical claims have numbered citations to authoritative sources
3. Follow the required structure: overview → theory → implementation → examples → applications
4. Include exercises to reinforce learning
5. Run validation scripts before submitting changes