# Data Model: Docusaurus Frontend Fixes for Humanoid Robotics Book

**Feature**: 1-humanoid-robotics-book | **Date**: 2025-12-14

## Overview

This data model describes the static assets and configuration elements that need to be created or modified to fix the Docusaurus frontend issues.

## Static Assets

### Asset: Favicon
- **Name**: favicon.ico
- **Type**: Image (ICO format)
- **Location**: static/img/favicon.ico
- **Purpose**: Browser tab icon and bookmarks
- **Dimensions**: 16x16, 32x32, or 48x48 pixels

### Asset: Navbar Logo
- **Name**: logo.svg
- **Type**: Image (SVG format)
- **Location**: static/img/logo.svg
- **Purpose**: Logo displayed in navbar
- **Dimensions**: Should be scalable vector graphic

### Asset: Social Card Image
- **Name**: docusaurus-social-card.jpg
- **Type**: Image (JPG format)
- **Location**: static/img/docusaurus-social-card.jpg
- **Purpose**: Image used for social media sharing
- **Dimensions**: Recommended 1200x630 pixels

## Configuration Elements

### Homepage Configuration
- **File**: docs/intro.md (frontmatter)
- **Field**: slug
- **Value**: "/"
- **Purpose**: Makes intro page serve as homepage

### Navigation Configuration
- **File**: docusaurus.config.js
- **Field**: themeConfig.navbar.logo.src
- **Value**: "img/logo.svg"
- **Purpose**: Points to navbar logo asset

## Validation Criteria

All assets must:
- Load without 404 errors
- Display properly in browser
- Be accessible via the configured baseUrl
- Pass Docusaurus build process