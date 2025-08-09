"""
Tests for prompt_templates.py module.

Tests the PromptType enum, PromptTemplates class functionality,
template variable validation, and prompt retrieval.
"""

import pytest
from unittest.mock import patch
from tubesensei.app.ai.prompt_templates import PromptType, PromptTemplates


class TestPromptType:
    """Test PromptType enum values."""
    
    def test_prompt_type_enum_values(self):
        """Test that all expected prompt types exist."""
        expected_types = [
            "video_filter",
            "idea_extraction", 
            "idea_categorization",
            "quality_assessment",
            "summary_generation"
        ]
        
        actual_types = [pt.value for pt in PromptType]
        
        assert set(expected_types) == set(actual_types)
        assert len(PromptType) == 5
    
    def test_prompt_type_accessibility(self):
        """Test that all prompt types are accessible."""
        assert PromptType.VIDEO_FILTER.value == "video_filter"
        assert PromptType.IDEA_EXTRACTION.value == "idea_extraction"
        assert PromptType.IDEA_CATEGORIZATION.value == "idea_categorization"
        assert PromptType.QUALITY_ASSESSMENT.value == "quality_assessment"
        assert PromptType.SUMMARY_GENERATION.value == "summary_generation"


class TestPromptTemplates:
    """Test PromptTemplates class functionality."""
    
    def test_system_prompts_exist_for_all_types(self):
        """Test that system prompts exist for all prompt types."""
        for prompt_type in PromptType:
            assert prompt_type in PromptTemplates.SYSTEM_PROMPTS
            system_prompt = PromptTemplates.SYSTEM_PROMPTS[prompt_type]
            assert isinstance(system_prompt, str)
            assert len(system_prompt) > 0
    
    def test_templates_exist_for_all_types(self):
        """Test that templates exist for all prompt types."""
        for prompt_type in PromptType:
            assert prompt_type in PromptTemplates.TEMPLATES
            template = PromptTemplates.TEMPLATES[prompt_type]
            assert isinstance(template, str)
            assert len(template) > 0
    
    def test_get_prompt_with_valid_variables_video_filter(self):
        """Test get_prompt with valid variables for video filter."""
        variables = {
            "title": "Test Video Title",
            "description": "Test description",
            "channel_name": "Test Channel",
            "duration_minutes": 15,
            "view_count": 1000,
            "published_date": "2024-01-01"
        }
        
        system_prompt, user_prompt = PromptTemplates.get_prompt(
            PromptType.VIDEO_FILTER, variables, include_system=True
        )
        
        assert system_prompt is not None
        assert "content analyst" in system_prompt.lower()
        assert user_prompt is not None
        assert "Test Video Title" in user_prompt
        assert "Test description" in user_prompt
        assert "Test Channel" in user_prompt
        assert "15 minutes" in user_prompt
    
    def test_get_prompt_with_valid_variables_idea_extraction(self):
        """Test get_prompt with valid variables for idea extraction."""
        variables = {
            "transcript": "This is a test transcript with business ideas.",
            "title": "Business Ideas Video",
            "channel_name": "Business Channel", 
            "duration_minutes": 20
        }
        
        system_prompt, user_prompt = PromptTemplates.get_prompt(
            PromptType.IDEA_EXTRACTION, variables
        )
        
        assert system_prompt is not None
        assert "business analyst" in system_prompt.lower()
        assert user_prompt is not None
        assert "This is a test transcript" in user_prompt
        assert "Business Ideas Video" in user_prompt
        assert "Business Channel" in user_prompt
    
    def test_get_prompt_with_valid_variables_quality_assessment(self):
        """Test get_prompt with valid variables for quality assessment."""
        variables = {
            "title": "Test Business Idea",
            "description": "A revolutionary business concept",
            "category": "SaaS",
            "source_context": "From startup podcast"
        }
        
        system_prompt, user_prompt = PromptTemplates.get_prompt(
            PromptType.QUALITY_ASSESSMENT, variables
        )
        
        assert system_prompt is not None
        assert "venture capitalist" in system_prompt.lower()
        assert user_prompt is not None
        assert "Test Business Idea" in user_prompt
        assert "A revolutionary business concept" in user_prompt
        assert "SaaS" in user_prompt
    
    def test_get_prompt_without_system_prompt(self):
        """Test get_prompt with include_system=False."""
        variables = {
            "title": "Test",
            "description": "Test desc",
            "channel_name": "Test Channel",
            "duration_minutes": 10,
            "view_count": 100,
            "published_date": "2024-01-01"
        }
        
        system_prompt, user_prompt = PromptTemplates.get_prompt(
            PromptType.VIDEO_FILTER, variables, include_system=False
        )
        
        assert system_prompt is None
        assert user_prompt is not None
        assert "Test" in user_prompt
    
    def test_get_prompt_with_missing_variables_raises_error(self):
        """Test that missing variables raise ValueError."""
        incomplete_variables = {
            "title": "Test Video",
            "description": "Test description"
            # Missing required variables
        }
        
        with pytest.raises(ValueError) as exc_info:
            PromptTemplates.get_prompt(PromptType.VIDEO_FILTER, incomplete_variables)
        
        assert "Missing required variable" in str(exc_info.value)
    
    def test_get_prompt_with_unknown_prompt_type_raises_error(self):
        """Test that unknown prompt type raises ValueError."""
        # Create a mock enum value that doesn't exist in templates
        with patch('tubesensei.app.ai.prompt_templates.PromptType') as mock_type:
            mock_type.UNKNOWN = "unknown_type"
            
            with pytest.raises(ValueError) as exc_info:
                PromptTemplates.get_prompt(mock_type.UNKNOWN, {})
            
            assert "Unknown prompt type" in str(exc_info.value)
    
    def test_validate_variables_with_complete_variables(self):
        """Test validate_variables with all required variables."""
        variables = {
            "title": "Test",
            "description": "Test desc", 
            "category": "SaaS",
            "source_context": "Test context"
        }
        
        result = PromptTemplates.validate_variables(
            PromptType.QUALITY_ASSESSMENT, variables
        )
        
        assert result is True
    
    def test_validate_variables_with_missing_variables_raises_error(self):
        """Test validate_variables with missing variables raises ValueError."""
        incomplete_variables = {
            "title": "Test",
            "description": "Test desc"
            # Missing category and source_context
        }
        
        with pytest.raises(ValueError) as exc_info:
            PromptTemplates.validate_variables(
                PromptType.QUALITY_ASSESSMENT, incomplete_variables
            )
        
        error_msg = str(exc_info.value)
        assert "Missing required variables" in error_msg
        assert "quality_assessment" in error_msg
    
    def test_get_required_variables_video_filter(self):
        """Test get_required_variables for video filter."""
        required_vars = PromptTemplates.get_required_variables(PromptType.VIDEO_FILTER)
        
        expected_vars = {
            "title", "description", "channel_name", 
            "duration_minutes", "view_count", "published_date"
        }
        
        assert required_vars == expected_vars
    
    def test_get_required_variables_idea_extraction(self):
        """Test get_required_variables for idea extraction."""
        required_vars = PromptTemplates.get_required_variables(PromptType.IDEA_EXTRACTION)
        
        expected_vars = {
            "transcript", "title", "channel_name", "duration_minutes"
        }
        
        assert required_vars == expected_vars
    
    def test_get_required_variables_quality_assessment(self):
        """Test get_required_variables for quality assessment."""
        required_vars = PromptTemplates.get_required_variables(PromptType.QUALITY_ASSESSMENT)
        
        expected_vars = {
            "title", "description", "category", "source_context"
        }
        
        assert required_vars == expected_vars
    
    def test_list_prompt_types(self):
        """Test list_prompt_types returns all available types."""
        prompt_types = PromptTemplates.list_prompt_types()
        
        assert isinstance(prompt_types, list)
        assert len(prompt_types) == 5
        
        # Check all expected types are present
        values = [pt.value for pt in prompt_types]
        expected_values = [
            "video_filter", "idea_extraction", "idea_categorization",
            "quality_assessment", "summary_generation"
        ]
        
        assert set(values) == set(expected_values)
    
    def test_get_prompt_info_complete(self):
        """Test get_prompt_info returns complete information."""
        info = PromptTemplates.get_prompt_info(PromptType.VIDEO_FILTER)
        
        assert isinstance(info, dict)
        assert info["type"] == "video_filter"
        assert info["has_system_prompt"] is True
        assert info["has_user_template"] is True
        assert isinstance(info["required_variables"], list)
        assert len(info["required_variables"]) > 0
        assert info["system_prompt_preview"] is not None
        assert len(info["system_prompt_preview"]) > 0
    
    def test_all_prompt_types_return_valid_templates(self):
        """Test that all prompt types return valid formatted templates."""
        # Sample variables that should work for all templates
        base_variables = {
            "title": "Test Title",
            "description": "Test description", 
            "channel_name": "Test Channel",
            "duration_minutes": 15,
            "view_count": 1000,
            "published_date": "2024-01-01",
            "transcript": "Test transcript content",
            "category": "SaaS",
            "source_context": "Test context",
            "video_title": "Video Title",
            "source": "Test Source",
            "idea_count": 5,
            "ideas_summary": "Test ideas",
            "key_insights": "Test insights"
        }
        
        for prompt_type in PromptType:
            # Get required variables for this prompt type
            required_vars = PromptTemplates.get_required_variables(prompt_type)
            
            # Create variables dict with only required variables
            test_variables = {
                var: base_variables.get(var, f"test_{var}") 
                for var in required_vars
            }
            
            # Test that get_prompt works without errors
            system_prompt, user_prompt = PromptTemplates.get_prompt(
                prompt_type, test_variables
            )
            
            assert system_prompt is not None
            assert user_prompt is not None
            assert len(system_prompt) > 0
            assert len(user_prompt) > 0
            
            # Check that no template variables remain unformatted
            assert "{" not in user_prompt
            assert "}" not in user_prompt
    
    def test_template_variable_extraction_regex(self):
        """Test that variable extraction regex works correctly."""
        test_template = "Hello {name}, you have {count} messages in {folder}."
        
        import re
        variables = set(re.findall(r'\{(\w+)\}', test_template))
        
        expected = {"name", "count", "folder"}
        assert variables == expected
    
    def test_validate_variables_with_empty_template(self):
        """Test validate_variables behavior with empty template."""
        # Mock empty template
        with patch.object(PromptTemplates, 'TEMPLATES', {PromptType.VIDEO_FILTER: ""}):
            result = PromptTemplates.validate_variables(
                PromptType.VIDEO_FILTER, {"any": "variable"}
            )
            assert result is True  # No variables required for empty template