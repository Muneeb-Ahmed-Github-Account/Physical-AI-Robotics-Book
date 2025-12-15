# Data Model: Homepage Branding & Visual Design Enhancements

**Feature**: 1-humanoid-robotics-book | **Date**: 2025-12-14

## Overview

This data model describes the visual assets, configuration elements, and component structures that need to be created or modified to implement the branding and visual design enhancements.

## Visual Assets

### Asset: Light Mode Logo
- **Name**: logo.svg
- **Type**: Vector image (SVG format)
- **Location**: static/img/logo.svg
- **Purpose**: Primary logo displayed in navbar in light mode
- **Dimensions**: 150-200px width, under 50px height
- **Optimization**: Minified SVG with unnecessary metadata removed

### Asset: Dark Mode Logo
- **Name**: logo-dark.svg
- **Type**: Vector image (SVG format)
- **Location**: static/img/logo-dark.svg
- **Purpose**: Logo displayed in navbar when dark mode is active
- **Dimensions**: Same as light mode logo
- **Design**: Adjusted colors for better visibility in dark mode

### Asset: Hero Background Image
- **Name**: hero-background.jpg
- **Type**: Raster image (JPG format)
- **Location**: static/img/hero-background.jpg
- **Purpose**: Background image for homepage hero section
- **Dimensions**: Optimized for web, responsive design
- **Quality**: Compressed for fast loading while maintaining visual quality

## Configuration Elements

### Navbar Logo Configuration
- **File**: docusaurus.config.js
- **Field**: themeConfig.navbar.logo
- **Properties**:
  - src: "img/logo.svg" (light mode logo)
  - srcDark: "img/logo-dark.svg" (dark mode logo)
  - alt: "Humanoid Robotics Logo"
  - href: "/" (link destination)
  - width: (optional, specify dimensions)
  - height: (optional, specify dimensions)

### Homepage Component Structure
- **File**: src/pages/index.js
- **Component**: Home
- **Structure**:
  - Layout: Main layout component with title and description
  - Hero Section: Container with background, centered content, and call-to-action
  - Content Elements: Title, tagline, and navigation button

## CSS Styling Elements

### Custom Styles
- **File**: src/css/custom.css
- **Classes**:
  - hero-banner: Main hero section styling
  - hero-title: Title text styling
  - hero-subtitle: Tagline text styling
  - hero-button: Call-to-action button styling
  - hero-background: Background image and effects

## Validation Criteria

All enhancements must:
- Maintain accessibility standards (WCAG 2.1 AA)
- Pass performance tests (good Lighthouse scores)
- Display correctly in both light and dark modes
- Be responsive across different screen sizes
- Maintain fast loading times
- Preserve the professional academic tone
- Work consistently across different browsers