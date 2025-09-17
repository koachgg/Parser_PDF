# Changelog

All notable changes to the PDF Parser project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Visual debugging functionality with color-coded content visualization
  - Added visual_debug.py module for bounding box visualization
  - Added configuration options in config.yaml for visual debugging
  - Added command-line flag --visual-debug to enable feature
  - Created comprehensive documentation in docs/visual_debugging.md
  - Added demonstration scripts in scripts/run_with_visual_debug.py
- Configuration management using YAML files
  - Added ConfigManager class for handling settings
  - Added default configuration with various options
  - Command-line arguments now override configuration file
- Enhanced content overlap detection and removal
  - Added content_overlap.py module to identify and resolve overlaps
  - Added intelligent table caption preservation
  - Added configuration options for overlap thresholds
  - Created comprehensive documentation in docs/content_overlap.md
  - Added unit tests for overlap handling functionality

### Changed
- Enhanced section tracking with improved heading detection
- Improved text block assembly with horizontal alignment consideration
- Enhanced table extraction with IoU-based deduplication
- Improved chart detection with vector graphics density analysis

### Fixed
- Fixed duplicate table detection issue
- Improved header/footer detection
- Fixed issues with text extraction in multi-column layouts
- Added proper error handling for missing dependencies
