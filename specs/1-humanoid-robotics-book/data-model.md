# Data Model: Humanoid Robotics & Physical AI Course Book

**Feature**: 1-humanoid-robotics-book
**Date**: 2025-12-10

## Core Entities

### Course Book
- **Description**: The complete educational material covering humanoid robotics and physical AI concepts
- **Attributes**:
  - title: string (Humanoid Robotics & Physical AI Course Book)
  - version: string (edition/version of the book)
  - chapters: array of Chapter objects
  - target_audience: string (intermediate-advanced engineering students)
  - course_duration: string (13-week course)
  - prerequisites: array of string (required background knowledge)

### Chapter
- **Description**: Individual sections of the book focused on specific topics
- **Attributes**:
  - id: string (unique identifier for the chapter)
  - title: string (chapter title)
  - module: string (associated course module: ROS2, Digital Twin, NVIDIA Isaac, VLA, Capstone)
  - structure: object (overview, theory, implementation, examples, applications)
  - content: string (main content in Markdown format)
  - code_examples: array of CodeExample objects
  - diagrams: array of Diagram objects
  - citations: array of Citation objects
  - learning_objectives: array of string
  - exercises: array of Exercise objects

### CodeExample
- **Description**: Code snippets included in chapters for practical learning
- **Attributes**:
  - id: string (unique identifier)
  - title: string (description of the example)
  - language: string (programming language)
  - code: string (actual code content)
  - explanation: string (description of what the code does)
  - validation_status: enum (unverified, verified, needs_update)

### Diagram
- **Description**: Visual representations to support understanding
- **Attributes**:
  - id: string (unique identifier)
  - title: string (description of the diagram)
  - type: string (block, flowchart, architecture, etc.)
  - description: string (explanation of the diagram)
  - file_path: string (path to the diagram file)

### Citation
- **Description**: References to authoritative sources
- **Attributes**:
  - id: string (unique identifier)
  - number: integer (citation number in the sequence)
  - type: enum (book, paper, documentation, website)
  - title: string (title of the source)
  - authors: array of string (authors of the source)
  - publication: string (journal, conference, or publisher)
  - year: integer (publication year)
  - url: string (optional URL to the source)

### Exercise
- **Description**: Practical tasks for students to complete
- **Attributes**:
  - id: string (unique identifier)
  - title: string (exercise title)
  - type: enum (theoretical, implementation, analysis, research)
  - difficulty: enum (beginner, intermediate, advanced)
  - description: string (detailed description of the exercise)
  - expected_outcome: string (what the student should achieve)
  - hints: array of string (optional hints for the student)

### HardwareComponent
- **Description**: Physical components referenced in the hardware guide
- **Attributes**:
  - id: string (unique identifier)
  - name: string (component name)
  - category: enum (workstation, embedded, sensor, robot_platform)
  - specifications: object (technical specifications)
  - use_case: string (how the component is used)
  - alternatives: array of string (alternative components)
  - cost_range: string (approximate cost range)

## Relationships

- Course Book "contains" multiple Chapters
- Chapter "contains" multiple CodeExamples, Diagrams, Citations, and Exercises
- Chapter "references" multiple HardwareComponents (in hardware guide chapters)
- Citation "supports" specific claims within Chapters

## Validation Rules

- Each Chapter must follow the structure: overview → theory → implementation → examples → applications
- Each Citation must have a valid reference to authoritative sources
- Each CodeExample must be validated for correctness
- Each Chapter must have at least one Exercise
- Content must be appropriate for intermediate-advanced engineering audience
- All technical claims must be supported by Citations