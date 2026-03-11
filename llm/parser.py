"""
Parser for validating TaskSpecification objects against system context.
"""

import re
from typing import Optional
from datetime import datetime, timedelta, timezone

from .interface import (
    TaskSpecification,
    LLMParseError
)


class TaskSpecificationParser:
    """
    Validates TaskSpecification objects against system context.
    Pydantic validation and JSON parsing is handled by instructor.
    """
    
    @staticmethod
    def validate_against_context(
        task_spec: TaskSpecification,
        available_sensors: list[str],
        available_locations: list[str],
        time_range: tuple[datetime, datetime]
    ) -> list[str]:
        """
        Validate task specification against system context.
        Returns list of validation errors (empty if valid).
        
        Args:
            task_spec: Parsed task specification
            available_sensors: Valid sensor types
            available_locations: Valid location identifiers
            time_range: (min_date, max_date) for available data
            
        Returns:
            List of validation error messages
        """
        errors = []
        
        # Validate sensor type
        if task_spec.sensor_type not in available_sensors:
            errors.append(
                f"Unknown sensor type '{task_spec.sensor_type}'. "
                f"Available sensors: {', '.join(available_sensors)}"
            )
        
        # Validate locations
        locations = task_spec.get_locations_list()
        invalid_locations = [loc for loc in locations if loc not in available_locations]
        
        if invalid_locations:
            # Show a few suggestions if there are many locations
            suggestions = available_locations[:5]
            errors.append(
                f"Unknown location(s): {', '.join(invalid_locations)}. "
                f"Available locations include: {', '.join(suggestions)}..."
            )
        
        # Validate time range
        min_date, max_date = time_range
        if task_spec.start_time.date() < min_date.date():
            errors.append(
                f"Start time ({task_spec.start_time.date()}) is before available data. "
                f"Data starts from {min_date.date()}."
            )
        if task_spec.end_time.date() > max_date.date():
            errors.append(
                f"End time ({task_spec.end_time.date()}) is after available data. "
                f"Data available until {max_date.date()}."
            )
        
        # Validate time range makes sense (should be caught by Pydantic, but double-check)
        if task_spec.end_time <= task_spec.start_time:
            errors.append("End time must be after start time.")
        
        return errors


class RelativeDateParser:
    """Helper class for parsing relative date expressions."""
    
    @staticmethod
    def parse_relative_date(
        expression: str,
        reference_date: Optional[datetime] = None
    ) -> tuple[datetime, datetime]:
        """
        Parse relative date expressions like "today", "yesterday", "last week".
        
        Args:
            expression: Relative date expression
            reference_date: Reference date (default: now)
            
        Returns:
            Tuple of (start_datetime, end_datetime)
        """
        if reference_date is None:
            reference_date = datetime.now(timezone.utc)
        
        expression = expression.lower().strip()
        
        # Today
        if expression == "today":
            start = reference_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start.replace(hour=23, minute=59, second=59)
            return start, end
        
        # Yesterday
        if expression == "yesterday":
            yesterday = reference_date - timedelta(days=1)
            start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            end = start.replace(hour=23, minute=59, second=59)
            if end > reference_date:
                end = reference_date
            return start, end
        
        # This week
        if expression in ["this week", "current week"]:
            # Start of week (Monday)
            days_since_monday = reference_date.weekday()
            start = (reference_date - timedelta(days=days_since_monday)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            end = reference_date.replace(hour=23, minute=59, second=59)
            return start, end
        
        # Last week
        if expression == "last week":
            end = reference_date.replace(hour=23, minute=59, second=59)
            start = (reference_date - timedelta(days=7)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            return start, end
        
        # Last month
        if expression == "last month":
            end = reference_date.replace(hour=23, minute=59, second=59)
            start = (reference_date - timedelta(days=30)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            return start, end
        
        # Last N days
        match = re.match(r'last (\d+) days?', expression)
        if match:
            days = int(match.group(1))
            end = reference_date.replace(hour=23, minute=59, second=59)
            start = (reference_date - timedelta(days=days)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            return start, end
        
        # Past N hours
        match = re.match(r'past (\d+) hours?', expression)
        if match:
            hours = int(match.group(1))
            end = reference_date
            start = reference_date - timedelta(hours=hours)
            return start, end
        
        raise ValueError(f"Cannot parse relative date expression: '{expression}'")