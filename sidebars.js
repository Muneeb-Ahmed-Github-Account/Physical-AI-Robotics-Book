// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  tutorialSidebar: [
    'intro',
    'course-summary',
    'assessment-rubrics',
    {
      type: 'category',
      label: 'ROS 2 Fundamentals',
      items: [
        'ros2/index',
        'ros2/overview',
        'ros2/theory',
        'ros2/implementation',
        'ros2/examples',
        'ros2/applications',
        'ros2/exercises',
        'ros2/diagrams',
        'ros2/testing',
      ],
    },
    {
      type: 'category',
      label: 'Digital Twin Simulation',
      items: [
        'digital-twin/index',
        'digital-twin/gazebo-unity',
        'digital-twin/simulation-basics',
        'digital-twin/advanced-sim',
        'digital-twin/integration',
        'digital-twin/exercises',
        'digital-twin/diagrams',
        'digital-twin/testing',
        'digital-twin/instructor-notes',
      ],
    },
    {
      type: 'category',
      label: 'NVIDIA Isaac',
      items: [
        'nvidia-isaac/index',
        'nvidia-isaac/setup',
        'nvidia-isaac/core-concepts',
        'nvidia-isaac/examples',
        'nvidia-isaac/best-practices',
      ],
    },
    {
      type: 'category',
      label: 'Vision-Language-Action Systems',
      items: [
        'vla-systems/index',
        'vla-systems/overview',
        'vla-systems/architecture',
        'vla-systems/implementation',
        'vla-systems/applications',
      ],
    },
    {
      type: 'category',
      label: 'Capstone Humanoid Project',
      items: [
        'capstone-humanoid/index',
        'capstone-humanoid/project-outline',
        'capstone-humanoid/implementation',
        'capstone-humanoid/testing',
        'capstone-humanoid/deployment',
      ],
    },
    {
      type: 'category',
      label: 'Hardware Guide',
      items: [
        'hardware-guide/index',
        'hardware-guide/workstation-setup',
        'hardware-guide/jetson-kits',
        'hardware-guide/sensors',
        'hardware-guide/robot-options',
      ],
    },
  ],
};

export default sidebars;