# TubeSensei Phase 3D: Export & Integration
## Week 9 - Days 4-5: Export Systems and Documentation

### Version: 1.0
### Duration: 2 Days
### Dependencies: Phase 3A, 3B, 3C Complete

---

## Table of Contents
1. [Phase Overview](#phase-overview)
2. [Day 4: Export Systems](#day-4-export-systems)
3. [Day 5: API Documentation & Testing](#day-5-api-documentation--testing)
4. [Implementation Checklist](#implementation-checklist)
5. [Testing Requirements](#testing-requirements)

---

## Phase Overview

### Objectives
Complete the TubeSensei platform with comprehensive export capabilities, webhook system, API documentation, and integration tools.

### Deliverables
- Multi-format export system (JSON, CSV, Excel, IdeaHunter)
- Webhook notification system
- Complete OpenAPI documentation
- SDK generation and examples
- Integration guides
- Performance optimization
- End-to-end testing

### Integration Focus
- IdeaHunter compatibility
- Third-party webhook support
- API client libraries
- Documentation portal

---

## Day 4: Export Systems

### 4.1 Export Service Implementation

```python
# app/services/export_service.py
from typing import List, Dict, Any, Optional
from datetime import datetime
from io import BytesIO
import json
import csv
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from zipfile import ZipFile
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from app.models.idea import Idea
from app.models.video import Video
from app.models.channel import Channel
from app.models.transcript import Transcript
from app.core.exceptions import ValidationException

class ExportService:
    """Comprehensive export service for multiple formats"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def export_ideas(
        self,
        idea_ids: List[str],
        format: str,
        options: Dict[str, Any] = None
    ) -> BytesIO:
        """
        Export ideas in specified format.
        
        Formats:
            - json: Standard JSON
            - csv: CSV spreadsheet
            - excel: Excel workbook with formatting
            - ideahunter: IdeaHunter compatible format
            - markdown: Markdown documentation
            - zip: Complete export with all formats
        """
        options = options or {}
        
        # Fetch ideas with related data
        ideas = await self._fetch_ideas_with_context(idea_ids)
        
        if format == "json":
            return await self._export_json(ideas, options)
        elif format == "csv":
            return await self._export_csv(ideas, options)
        elif format == "excel":
            return await self._export_excel(ideas, options)
        elif format == "ideahunter":
            return await self._export_ideahunter(ideas, options)
        elif format == "markdown":
            return await self._export_markdown(ideas, options)
        elif format == "zip":
            return await self._export_zip(ideas, options)
        else:
            raise ValidationException({"format": f"Unsupported format: {format}"})
    
    async def _fetch_ideas_with_context(
        self,
        idea_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Fetch ideas with full context"""
        ideas_data = []
        
        for idea_id in idea_ids:
            # Get idea
            idea = await self.db.get(Idea, idea_id)
            if not idea:
                continue
            
            # Get video
            video = await self.db.get(Video, idea.video_id)
            
            # Get channel
            channel = await self.db.get(Channel, video.channel_id) if video else None
            
            # Get transcript excerpt
            transcript_result = await self.db.execute(
                select(Transcript).where(Transcript.video_id == video.id)
            )
            transcript = transcript_result.scalar_one_or_none()
            
            ideas_data.append({
                "idea": idea,
                "video": video,
                "channel": channel,
                "transcript_excerpt": self._get_relevant_excerpt(
                    transcript,
                    idea.source_timestamp
                ) if transcript else None
            })
        
        return ideas_data
    
    async def _export_json(
        self,
        ideas: List[Dict[str, Any]],
        options: Dict[str, Any]
    ) -> BytesIO:
        """Export as JSON"""
        include_context = options.get("include_context", True)
        pretty = options.get("pretty", True)
        
        export_data = {
            "export_date": datetime.utcnow().isoformat(),
            "export_version": "1.0",
            "total_ideas": len(ideas),
            "ideas": []
        }
        
        for item in ideas:
            idea = item["idea"]
            idea_data = {
                "id": str(idea.id),
                "title": idea.title,
                "description": idea.description,
                "category": idea.category,
                "confidence_score": idea.confidence_score,
                "complexity_score": idea.complexity_score,
                "market_size_estimate": idea.market_size_estimate,
                "tags": idea.tags,
                "status": idea.status,
                "created_at": idea.created_at.isoformat()
            }
            
            if include_context:
                video = item["video"]
                channel = item["channel"]
                
                idea_data["source"] = {
                    "video": {
                        "id": str(video.id),
                        "title": video.title,
                        "url": video.video_url,
                        "published_at": video.published_at.isoformat(),
                        "duration": video.duration_seconds,
                        "views": video.view_count
                    } if video else None,
                    "channel": {
                        "id": str(channel.id),
                        "name": channel.name,
                        "url": channel.channel_url,
                        "subscribers": channel.subscriber_count
                    } if channel else None,
                    "transcript_excerpt": item.get("transcript_excerpt")
                }
            
            export_data["ideas"].append(idea_data)
        
        # Convert to BytesIO
        buffer = BytesIO()
        json_str = json.dumps(export_data, indent=2 if pretty else None)
        buffer.write(json_str.encode())
        buffer.seek(0)
        
        return buffer
    
    async def _export_csv(
        self,
        ideas: List[Dict[str, Any]],
        options: Dict[str, Any]
    ) -> BytesIO:
        """Export as CSV"""
        include_source = options.get("include_source", True)
        
        rows = []
        for item in ideas:
            idea = item["idea"]
            video = item["video"]
            channel = item["channel"]
            
            row = {
                "ID": str(idea.id),
                "Title": idea.title,
                "Description": idea.description,
                "Category": idea.category or "",
                "Confidence Score": f"{idea.confidence_score:.2%}",
                "Complexity (1-10)": idea.complexity_score,
                "Market Size": idea.market_size_estimate or "",
                "Tags": ", ".join(idea.tags) if idea.tags else "",
                "Status": idea.status,
                "Created At": idea.created_at.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            if include_source:
                row.update({
                    "Video Title": video.title if video else "",
                    "Video URL": video.video_url if video else "",
                    "Channel": channel.name if channel else "",
                    "Published Date": video.published_at.strftime("%Y-%m-%d") if video else ""
                })
            
            rows.append(row)
        
        # Create CSV
        buffer = BytesIO()
        wrapper = io.TextIOWrapper(buffer, encoding='utf-8', newline='')
        
        if rows:
            writer = csv.DictWriter(wrapper, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        
        wrapper.flush()
        buffer.seek(0)
        
        return buffer
    
    async def _export_excel(
        self,
        ideas: List[Dict[str, Any]],
        options: Dict[str, Any]
    ) -> BytesIO:
        """Export as formatted Excel workbook"""
        wb = Workbook()
        
        # Ideas sheet
        ws_ideas = wb.active
        ws_ideas.title = "Ideas"
        
        # Define headers
        headers = [
            "ID", "Title", "Description", "Category",
            "Confidence", "Complexity", "Market Size",
            "Tags", "Status", "Created Date",
            "Video Title", "Channel", "Video URL"
        ]
        
        # Style headers
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_alignment = Alignment(horizontal="center", vertical="center")
        
        for col, header in enumerate(headers, 1):
            cell = ws_ideas.cell(row=1, column=col, value=header)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment
        
        # Add data
        for row_idx, item in enumerate(ideas, 2):
            idea = item["idea"]
            video = item["video"]
            channel = item["channel"]
            
            ws_ideas.cell(row=row_idx, column=1, value=str(idea.id))
            ws_ideas.cell(row=row_idx, column=2, value=idea.title)
            ws_ideas.cell(row=row_idx, column=3, value=idea.description)
            ws_ideas.cell(row=row_idx, column=4, value=idea.category or "")
            ws_ideas.cell(row=row_idx, column=5, value=idea.confidence_score).number_format = "0.00%"
            ws_ideas.cell(row=row_idx, column=6, value=idea.complexity_score)
            ws_ideas.cell(row=row_idx, column=7, value=idea.market_size_estimate or "")
            ws_ideas.cell(row=row_idx, column=8, value=", ".join(idea.tags) if idea.tags else "")
            ws_ideas.cell(row=row_idx, column=9, value=idea.status)
            ws_ideas.cell(row=row_idx, column=10, value=idea.created_at.strftime("%Y-%m-%d"))
            ws_ideas.cell(row=row_idx, column=11, value=video.title if video else "")
            ws_ideas.cell(row=row_idx, column=12, value=channel.name if channel else "")
            ws_ideas.cell(row=row_idx, column=13, value=video.video_url if video else "")
        
        # Auto-adjust column widths
        for column in ws_ideas.columns:
            max_length = 0
            column_letter = column[0].column_letter
            
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            
            adjusted_width = min(max_length + 2, 50)
            ws_ideas.column_dimensions[column_letter].width = adjusted_width
        
        # Add summary sheet
        ws_summary = wb.create_sheet("Summary")
        
        # Summary statistics
        summary_data = [
            ["Metric", "Value"],
            ["Total Ideas", len(ideas)],
            ["Average Confidence", f"{sum(i['idea'].confidence_score for i in ideas) / len(ideas):.2%}" if ideas else "0%"],
            ["Categories", len(set(i["idea"].category for i in ideas if i["idea"].category))],
            ["Selected Ideas", sum(1 for i in ideas if i["idea"].status == "selected")],
            ["Export Date", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")]
        ]
        
        for row_idx, row_data in enumerate(summary_data, 1):
            for col_idx, value in enumerate(row_data, 1):
                cell = ws_summary.cell(row=row_idx, column=col_idx, value=value)
                if row_idx == 1:
                    cell.font = header_font
                    cell.fill = header_fill
        
        # Save to buffer
        buffer = BytesIO()
        wb.save(buffer)
        buffer.seek(0)
        
        return buffer
    
    async def _export_ideahunter(
        self,
        ideas: List[Dict[str, Any]],
        options: Dict[str, Any]
    ) -> BytesIO:
        """Export in IdeaHunter compatible format"""
        export_data = {
            "source": "TubeSensei",
            "export_version": "1.0",
            "export_date": datetime.utcnow().isoformat(),
            "metadata": {
                "total_ideas": len(ideas),
                "export_options": options
            },
            "ideas": []
        }
        
        for item in ideas:
            idea = item["idea"]
            video = item["video"]
            channel = item["channel"]
            
            idea_hunter_format = {
                "external_id": str(idea.id),
                "title": idea.title,
                "description": idea.description,
                "category": idea.category,
                "tags": idea.tags,
                "scores": {
                    "confidence": idea.confidence_score,
                    "complexity": idea.complexity_score,
                    "market_potential": self._estimate_market_potential(idea)
                },
                "metadata": {
                    "source_type": "youtube_video",
                    "source_url": video.video_url if video else None,
                    "source_title": video.title if video else None,
                    "source_channel": channel.name if channel else None,
                    "extraction_date": idea.created_at.isoformat(),
                    "market_size_estimate": idea.market_size_estimate
                },
                "research_notes": item.get("transcript_excerpt", ""),
                "status": "pending_research",
                "priority": self._calculate_priority(idea)
            }
            
            export_data["ideas"].append(idea_hunter_format)
        
        # Sort by priority
        export_data["ideas"].sort(key=lambda x: x["priority"], reverse=True)
        
        buffer = BytesIO()
        json_str = json.dumps(export_data, indent=2)
        buffer.write(json_str.encode())
        buffer.seek(0)
        
        return buffer
    
    async def _export_markdown(
        self,
        ideas: List[Dict[str, Any]],
        options: Dict[str, Any]
    ) -> BytesIO:
        """Export as Markdown documentation"""
        lines = []
        
        # Header
        lines.append("# TubeSensei Ideas Export")
        lines.append(f"\n**Export Date:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"**Total Ideas:** {len(ideas)}\n")
        
        # Table of Contents
        lines.append("## Table of Contents\n")
        for idx, item in enumerate(ideas, 1):
            idea = item["idea"]
            lines.append(f"{idx}. [{idea.title}](#{idx}-{idea.title.lower().replace(' ', '-')})")
        
        lines.append("\n---\n")
        
        # Ideas
        for idx, item in enumerate(ideas, 1):
            idea = item["idea"]
            video = item["video"]
            channel = item["channel"]
            
            lines.append(f"## {idx}. {idea.title}\n")
            
            # Metadata table
            lines.append("| Property | Value |")
            lines.append("|----------|-------|")
            lines.append(f"| **Category** | {idea.category or 'Uncategorized'} |")
            lines.append(f"| **Confidence** | {idea.confidence_score:.1%} |")
            lines.append(f"| **Complexity** | {idea.complexity_score}/10 |")
            lines.append(f"| **Market Size** | {idea.market_size_estimate or 'Unknown'} |")
            lines.append(f"| **Status** | {idea.status} |")
            
            if idea.tags:
                lines.append(f"| **Tags** | {', '.join(idea.tags)} |")
            
            lines.append("")
            
            # Description
            lines.append("### Description\n")
            lines.append(idea.description)
            lines.append("")
            
            # Source
            if video:
                lines.append("### Source\n")
                lines.append(f"- **Video:** [{video.title}]({video.video_url})")
                lines.append(f"- **Channel:** {channel.name if channel else 'Unknown'}")
                lines.append(f"- **Published:** {video.published_at.strftime('%Y-%m-%d')}")
                lines.append("")
            
            # Transcript excerpt
            if item.get("transcript_excerpt"):
                lines.append("### Relevant Excerpt\n")
                lines.append(f"> {item['transcript_excerpt']}")
                lines.append("")
            
            lines.append("---\n")
        
        # Footer
        lines.append("\n## Export Information\n")
        lines.append(f"- Generated by TubeSensei v1.0")
        lines.append(f"- Export format: Markdown")
        lines.append(f"- Ideas exported: {len(ideas)}")
        
        # Convert to buffer
        buffer = BytesIO()
        content = "\n".join(lines)
        buffer.write(content.encode())
        buffer.seek(0)
        
        return buffer
    
    async def _export_zip(
        self,
        ideas: List[Dict[str, Any]],
        options: Dict[str, Any]
    ) -> BytesIO:
        """Export as ZIP archive with multiple formats"""
        buffer = BytesIO()
        
        with ZipFile(buffer, 'w') as zipf:
            # Add JSON export
            json_buffer = await self._export_json(ideas, options)
            zipf.writestr("ideas.json", json_buffer.getvalue())
            
            # Add CSV export
            csv_buffer = await self._export_csv(ideas, options)
            zipf.writestr("ideas.csv", csv_buffer.getvalue())
            
            # Add Excel export
            excel_buffer = await self._export_excel(ideas, options)
            zipf.writestr("ideas.xlsx", excel_buffer.getvalue())
            
            # Add Markdown export
            md_buffer = await self._export_markdown(ideas, options)
            zipf.writestr("ideas.md", md_buffer.getvalue())
            
            # Add IdeaHunter export
            ih_buffer = await self._export_ideahunter(ideas, options)
            zipf.writestr("ideas_ideahunter.json", ih_buffer.getvalue())
            
            # Add README
            readme_content = self._generate_readme(ideas)
            zipf.writestr("README.txt", readme_content)
        
        buffer.seek(0)
        return buffer
    
    def _get_relevant_excerpt(
        self,
        transcript: Optional[Transcript],
        timestamp: Optional[int],
        context_words: int = 100
    ) -> Optional[str]:
        """Extract relevant portion of transcript"""
        if not transcript or not transcript.content:
            return None
        
        if not timestamp:
            # Return first portion if no timestamp
            words = transcript.content.split()[:context_words]
            return " ".join(words) + "..."
        
        # Find approximate position based on timestamp
        # This is simplified - in production would use proper timestamp parsing
        words = transcript.content.split()
        position = int((timestamp / 3600) * len(words))  # Rough estimate
        
        start = max(0, position - context_words // 2)
        end = min(len(words), position + context_words // 2)
        
        excerpt = " ".join(words[start:end])
        
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(words):
            excerpt = excerpt + "..."
        
        return excerpt
    
    def _estimate_market_potential(self, idea: Idea) -> float:
        """Estimate market potential score (0-1)"""
        score = idea.confidence_score * 0.4
        
        # Adjust based on complexity (simpler = higher potential)
        complexity_factor = (10 - idea.complexity_score) / 10
        score += complexity_factor * 0.3
        
        # Category bonus
        high_potential_categories = ["Technology", "AI", "Automation", "SaaS"]
        if idea.category in high_potential_categories:
            score += 0.3
        
        return min(1.0, score)
    
    def _calculate_priority(self, idea: Idea) -> int:
        """Calculate priority score (1-100)"""
        priority = int(idea.confidence_score * 50)
        
        # Boost for selected ideas
        if idea.status == "selected":
            priority += 30
        elif idea.status == "reviewed":
            priority += 10
        
        # Complexity adjustment
        if idea.complexity_score <= 5:
            priority += 10
        
        return min(100, priority)
    
    def _generate_readme(self, ideas: List[Dict[str, Any]]) -> str:
        """Generate README for ZIP export"""
        return f"""TubeSensei Ideas Export
========================

Export Date: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}
Total Ideas: {len(ideas)}

Files Included:
--------------
- ideas.json: Complete data in JSON format
- ideas.csv: Spreadsheet-compatible CSV format
- ideas.xlsx: Formatted Excel workbook
- ideas.md: Markdown documentation
- ideas_ideahunter.json: IdeaHunter compatible format

Categories Included:
-------------------
{', '.join(set(i['idea'].category for i in ideas if i['idea'].category))}

Status Breakdown:
----------------
- Selected: {sum(1 for i in ideas if i['idea'].status == 'selected')}
- Reviewed: {sum(1 for i in ideas if i['idea'].status == 'reviewed')}
- Extracted: {sum(1 for i in ideas if i['idea'].status == 'extracted')}

For more information, visit: https://tubesensei.example.com
"""
```

### 4.2 Export API Endpoints

```python
# app/api/v1/export.py
from fastapi import APIRouter, Depends, Query, Body, HTTPException
from fastapi.responses import StreamingResponse
from typing import List, Optional
from uuid import UUID
from datetime import datetime

from app.schemas.api_responses import APIResponse
from app.services.export_service import ExportService
from app.core.api_auth import require_api_key
from app.core.database import get_db

router = APIRouter()

@router.post(
    "/ideas",
    summary="Export ideas",
    description="Export selected ideas in various formats"
)
async def export_ideas(
    idea_ids: List[UUID] = Body(..., description="List of idea IDs to export"),
    format: str = Body("json", regex="^(json|csv|excel|ideahunter|markdown|zip)$"),
    include_context: bool = Body(True, description="Include source video and channel info"),
    include_transcript: bool = Body(False, description="Include transcript excerpts"),
    api_key = Depends(require_api_key),
    db = Depends(get_db)
):
    """
    Export ideas in specified format.
    
    **Formats:**
    - `json`: Standard JSON format
    - `csv`: CSV spreadsheet
    - `excel`: Formatted Excel workbook
    - `ideahunter`: IdeaHunter compatible format
    - `markdown`: Markdown documentation
    - `zip`: Archive with all formats
    
    **Options:**
    - `include_context`: Include video and channel information
    - `include_transcript`: Include relevant transcript excerpts
    """
    service = ExportService(db)
    
    options = {
        "include_context": include_context,
        "include_transcript": include_transcript,
        "include_source": True,
        "pretty": True
    }
    
    try:
        buffer = await service.export_ideas(
            idea_ids=[str(id) for id in idea_ids],
            format=format,
            options=options
        )
        
        # Determine content type and filename
        content_types = {
            "json": ("application/json", "ideas.json"),
            "csv": ("text/csv", "ideas.csv"),
            "excel": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "ideas.xlsx"),
            "ideahunter": ("application/json", "ideas_ideahunter.json"),
            "markdown": ("text/markdown", "ideas.md"),
            "zip": ("application/zip", "ideas_export.zip")
        }
        
        content_type, filename = content_types[format]
        
        return StreamingResponse(
            buffer,
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "X-Export-Count": str(len(idea_ids)),
                "X-Export-Date": datetime.utcnow().isoformat()
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Export failed: {str(e)}"
        )

@router.get(
    "/formats",
    response_model=APIResponse[dict],
    summary="Get available export formats"
)
async def get_export_formats(
    api_key = Depends(require_api_key)
):
    """Get information about available export formats."""
    formats = {
        "json": {
            "name": "JSON",
            "description": "Standard JSON format with full data",
            "mime_type": "application/json",
            "supports_context": True,
            "supports_bulk": True
        },
        "csv": {
            "name": "CSV",
            "description": "Comma-separated values for spreadsheets",
            "mime_type": "text/csv",
            "supports_context": False,
            "supports_bulk": True
        },
        "excel": {
            "name": "Excel",
            "description": "Formatted Excel workbook with multiple sheets",
            "mime_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "supports_context": True,
            "supports_bulk": True
        },
        "ideahunter": {
            "name": "IdeaHunter",
            "description": "IdeaHunter platform compatible format",
            "mime_type": "application/json",
            "supports_context": True,
            "supports_bulk": True
        },
        "markdown": {
            "name": "Markdown",
            "description": "Formatted documentation in Markdown",
            "mime_type": "text/markdown",
            "supports_context": True,
            "supports_bulk": True
        },
        "zip": {
            "name": "ZIP Archive",
            "description": "Complete export with all formats",
            "mime_type": "application/zip",
            "supports_context": True,
            "supports_bulk": True
        }
    }
    
    return APIResponse(data=formats)

@router.post(
    "/bulk",
    summary="Bulk export with filters",
    description="Export ideas based on filter criteria"
)
async def bulk_export(
    filters: dict = Body(..., description="Filter criteria"),
    format: str = Body("json"),
    limit: int = Body(1000, le=10000),
    api_key = Depends(require_api_key),
    db = Depends(get_db)
):
    """
    Export ideas matching filter criteria.
    
    **Filter Example:**
    ```json
    {
        "status": "selected",
        "min_confidence": 0.7,
        "categories": ["Technology", "AI"],
        "date_from": "2024-01-01",
        "date_to": "2024-12-31"
    }
    ```
    """
    from app.services.idea_service import IdeaService
    
    idea_service = IdeaService(db)
    export_service = ExportService(db)
    
    # Get ideas matching filters
    result = await idea_service.list_ideas(
        status=filters.get("status"),
        min_confidence=filters.get("min_confidence", 0),
        category=filters.get("category"),
        limit=limit
    )
    
    idea_ids = [item["id"] for item in result["items"]]
    
    if not idea_ids:
        raise HTTPException(
            status_code=404,
            detail="No ideas found matching the filters"
        )
    
    # Export
    buffer = await export_service.export_ideas(
        idea_ids=idea_ids,
        format=format,
        options={"include_context": True}
    )
    
    content_types = {
        "json": ("application/json", "filtered_ideas.json"),
        "csv": ("text/csv", "filtered_ideas.csv"),
        "excel": ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "filtered_ideas.xlsx")
    }
    
    content_type, filename = content_types.get(format, ("application/octet-stream", "export"))
    
    return StreamingResponse(
        buffer,
        media_type=content_type,
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "X-Export-Count": str(len(idea_ids))
        }
    )
```

### 4.3 Webhook System

```python
# app/services/webhook_service.py
from typing import Dict, Any, List, Optional
from datetime import datetime
import httpx
import asyncio
import json
import hmac
import hashlib
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.webhook import Webhook, WebhookEvent, WebhookDelivery
from app.core.exceptions import ValidationException

class WebhookService:
    """Service for managing and delivering webhooks"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def register_webhook(
        self,
        url: str,
        events: List[str],
        secret: Optional[str] = None,
        active: bool = True
    ) -> Webhook:
        """Register a new webhook"""
        # Validate URL
        if not url.startswith(("http://", "https://")):
            raise ValidationException({"url": "Invalid webhook URL"})
        
        # Validate events
        valid_events = [
            "idea.created",
            "idea.updated",
            "idea.selected",
            "video.processed",
            "channel.synced",
            "job.completed",
            "job.failed"
        ]
        
        for event in events:
            if event not in valid_events:
                raise ValidationException({"events": f"Invalid event: {event}"})
        
        webhook = Webhook(
            url=url,
            events=events,
            secret=secret or Webhook.generate_secret(),
            active=active
        )
        
        self.db.add(webhook)
        await self.db.commit()
        await self.db.refresh(webhook)
        
        return webhook
    
    async def trigger_event(
        self,
        event_type: str,
        payload: Dict[str, Any]
    ):
        """Trigger webhook event for all registered webhooks"""
        # Get active webhooks for this event
        result = await self.db.execute(
            select(Webhook).where(
                Webhook.active == True,
                Webhook.events.contains([event_type])
            )
        )
        webhooks = result.scalars().all()
        
        if not webhooks:
            return
        
        # Create event record
        event = WebhookEvent(
            event_type=event_type,
            payload=payload
        )
        self.db.add(event)
        await self.db.commit()
        
        # Deliver to each webhook
        tasks = []
        for webhook in webhooks:
            task = asyncio.create_task(
                self._deliver_webhook(webhook, event)
            )
            tasks.append(task)
        
        # Wait for all deliveries
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _deliver_webhook(
        self,
        webhook: Webhook,
        event: WebhookEvent
    ):
        """Deliver webhook to endpoint"""
        delivery = WebhookDelivery(
            webhook_id=webhook.id,
            event_id=event.id,
            attempt=1
        )
        
        payload = {
            "event": event.event_type,
            "timestamp": event.created_at.isoformat(),
            "data": event.payload
        }
        
        # Sign payload if secret exists
        headers = {
            "Content-Type": "application/json",
            "X-TubeSensei-Event": event.event_type,
            "X-TubeSensei-Delivery": str(delivery.id)
        }
        
        if webhook.secret:
            signature = self._generate_signature(webhook.secret, payload)
            headers["X-TubeSensei-Signature"] = signature
        
        try:
            response = await self.client.post(
                webhook.url,
                json=payload,
                headers=headers
            )
            
            delivery.response_status = response.status_code
            delivery.response_body = response.text[:1000]  # Store first 1000 chars
            delivery.delivered_at = datetime.utcnow()
            
            if response.status_code >= 200 and response.status_code < 300:
                delivery.success = True
            else:
                delivery.success = False
                delivery.error_message = f"HTTP {response.status_code}"
                
                # Retry logic
                if delivery.attempt < 3:
                    await self._schedule_retry(webhook, event, delivery.attempt + 1)
        
        except Exception as e:
            delivery.success = False
            delivery.error_message = str(e)
            
            # Retry logic
            if delivery.attempt < 3:
                await self._schedule_retry(webhook, event, delivery.attempt + 1)
        
        self.db.add(delivery)
        await self.db.commit()
    
    async def _schedule_retry(
        self,
        webhook: Webhook,
        event: WebhookEvent,
        attempt: int
    ):
        """Schedule webhook retry with exponential backoff"""
        delay = 2 ** attempt  # 2, 4, 8 seconds
        await asyncio.sleep(delay)
        
        delivery = WebhookDelivery(
            webhook_id=webhook.id,
            event_id=event.id,
            attempt=attempt
        )
        
        # Retry delivery
        await self._deliver_webhook(webhook, event)
    
    def _generate_signature(
        self,
        secret: str,
        payload: Dict[str, Any]
    ) -> str:
        """Generate HMAC signature for webhook payload"""
        payload_str = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            secret.encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"sha256={signature}"
    
    async def get_webhook_stats(
        self,
        webhook_id: str
    ) -> Dict[str, Any]:
        """Get delivery statistics for a webhook"""
        # Get total deliveries
        total_query = select(func.count(WebhookDelivery.id)).where(
            WebhookDelivery.webhook_id == webhook_id
        )
        total = await self.db.scalar(total_query)
        
        # Get successful deliveries
        success_query = select(func.count(WebhookDelivery.id)).where(
            WebhookDelivery.webhook_id == webhook_id,
            WebhookDelivery.success == True
        )
        successful = await self.db.scalar(success_query)
        
        # Get recent deliveries
        recent_query = select(WebhookDelivery).where(
            WebhookDelivery.webhook_id == webhook_id
        ).order_by(WebhookDelivery.created_at.desc()).limit(10)
        
        result = await self.db.execute(recent_query)
        recent = result.scalars().all()
        
        return {
            "total_deliveries": total,
            "successful_deliveries": successful,
            "success_rate": (successful / total * 100) if total > 0 else 0,
            "recent_deliveries": [
                {
                    "event_type": d.event.event_type,
                    "delivered_at": d.delivered_at.isoformat() if d.delivered_at else None,
                    "success": d.success,
                    "attempts": d.attempt,
                    "response_status": d.response_status
                }
                for d in recent
            ]
        }
```

---

## Day 5: API Documentation & Testing

### 5.1 OpenAPI Documentation Enhancement

```python
# app/core/openapi_customization.py
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from typing import Dict, Any

def customize_openapi(app: FastAPI) -> Dict[str, Any]:
    """Customize OpenAPI schema with enhanced documentation"""
    
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="TubeSensei API",
        version="1.0.0",
        description="""
# TubeSensei API Documentation

## Overview
TubeSensei is a powerful YouTube content analysis platform that automatically discovers, 
transcribes, and extracts business ideas from video content. This API provides programmatic 
access to all TubeSensei functionality.

## Authentication
All API endpoints require authentication via API key. Include your API key in the `X-API-Key` header:

```
X-API-Key: ts_your_api_key_here
```

## Rate Limiting
API requests are rate limited based on your tier:
- **Standard**: 100 requests/hour, 10,000 requests/day
- **Premium**: 1,000 requests/hour, 100,000 requests/day
- **Unlimited**: No rate limits

Rate limit information is included in response headers:
- `X-RateLimit-Limit`: Maximum requests per hour
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Unix timestamp when limit resets

## Response Format
All endpoints return JSON responses with consistent structure:

### Success Response
```json
{
    "success": true,
    "data": {...},
    "message": "Optional message",
    "timestamp": "2024-01-01T00:00:00Z"
}
```

### Error Response
```json
{
    "success": false,
    "error": "Error message",
    "error_code": "ERROR_CODE",
    "details": {...},
    "timestamp": "2024-01-01T00:00:00Z"
}
```

## Pagination
List endpoints support pagination with these parameters:
- `limit`: Items per page (max 1000)
- `offset`: Number of items to skip

## Webhooks
Register webhooks to receive real-time notifications for events:
- `idea.created`: New idea extracted
- `idea.selected`: Idea marked as selected
- `video.processed`: Video processing completed
- `job.completed`: Background job completed

## SDK and Code Examples
Official SDKs are available for:
- Python: `pip install tubesensei-sdk`
- JavaScript: `npm install @tubesensei/sdk`
- Go: `go get github.com/tubesensei/sdk-go`

## Support
For API support, contact: api-support@tubesensei.com
        """,
        routes=app.routes,
        servers=[
            {"url": "https://api.tubesensei.com", "description": "Production"},
            {"url": "https://staging-api.tubesensei.com", "description": "Staging"},
            {"url": "http://localhost:8000", "description": "Development"}
        ]
    )
    
    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for authentication"
        }
    }
    
    # Add global security
    openapi_schema["security"] = [{"APIKeyHeader": []}]
    
    # Add tags with descriptions
    openapi_schema["tags"] = [
        {
            "name": "channels",
            "description": "Manage YouTube channels for monitoring"
        },
        {
            "name": "videos",
            "description": "Access and process video content"
        },
        {
            "name": "ideas",
            "description": "Manage extracted business ideas"
        },
        {
            "name": "export",
            "description": "Export data in various formats"
        },
        {
            "name": "webhooks",
            "description": "Configure webhook notifications"
        },
        {
            "name": "search",
            "description": "Search across all content"
        }
    ]
    
    # Add example responses
    for path, methods in openapi_schema["paths"].items():
        for method, operation in methods.items():
            if method in ["get", "post", "put", "patch", "delete"]:
                # Add example responses
                if "responses" not in operation:
                    operation["responses"] = {}
                
                # Add common error responses
                operation["responses"]["401"] = {
                    "description": "Authentication required",
                    "content": {
                        "application/json": {
                            "example": {
                                "success": False,
                                "error": "API key required",
                                "error_code": "AUTH_REQUIRED"
                            }
                        }
                    }
                }
                
                operation["responses"]["429"] = {
                    "description": "Rate limit exceeded",
                    "content": {
                        "application/json": {
                            "example": {
                                "success": False,
                                "error": "Rate limit exceeded",
                                "error_code": "RATE_LIMIT"
                            }
                        }
                    }
                }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema
```

### 5.2 SDK Generation

```python
# scripts/generate_sdk.py
"""Generate client SDKs from OpenAPI schema"""

import json
import subprocess
from pathlib import Path

def generate_python_sdk():
    """Generate Python SDK using openapi-generator"""
    subprocess.run([
        "openapi-generator-cli", "generate",
        "-i", "openapi.json",
        "-g", "python",
        "-o", "./sdk/python",
        "--package-name", "tubesensei",
        "--project-name", "tubesensei-sdk"
    ])

def generate_typescript_sdk():
    """Generate TypeScript SDK"""
    subprocess.run([
        "openapi-generator-cli", "generate",
        "-i", "openapi.json",
        "-g", "typescript-axios",
        "-o", "./sdk/typescript",
        "--npm-name", "@tubesensei/sdk"
    ])

def generate_go_sdk():
    """Generate Go SDK"""
    subprocess.run([
        "openapi-generator-cli", "generate",
        "-i", "openapi.json",
        "-g", "go",
        "-o", "./sdk/go",
        "--package-name", "tubesensei"
    ])

if __name__ == "__main__":
    # Export OpenAPI schema
    from app.main import app
    from app.core.openapi_customization import customize_openapi
    
    schema = customize_openapi(app)
    
    with open("openapi.json", "w") as f:
        json.dump(schema, f, indent=2)
    
    # Generate SDKs
    generate_python_sdk()
    generate_typescript_sdk()
    generate_go_sdk()
    
    print("SDKs generated successfully!")
```

### 5.3 API Client Examples

```python
# examples/python_client.py
"""Example Python client for TubeSensei API"""

import asyncio
from tubesensei import TubeSenseiClient, Configuration

async def main():
    # Configure client
    config = Configuration(
        host="https://api.tubesensei.com",
        api_key={"X-API-Key": "your_api_key_here"}
    )
    
    client = TubeSenseiClient(configuration=config)
    
    # List channels
    channels = await client.channels.list_channels(limit=10)
    print(f"Found {channels.pagination.total} channels")
    
    # Search for ideas
    ideas = await client.ideas.list_ideas(
        min_confidence=0.8,
        category="Technology",
        limit=20
    )
    
    for idea in ideas.data:
        print(f"- {idea.title} (confidence: {idea.confidence_score:.1%})")
    
    # Export selected ideas
    selected_ideas = [idea.id for idea in ideas.data if idea.status == "selected"]
    
    if selected_ideas:
        export_result = await client.export.export_ideas(
            idea_ids=selected_ideas,
            format="json"
        )
        
        with open("exported_ideas.json", "wb") as f:
            f.write(export_result.content)
    
    # Register webhook
    webhook = await client.webhooks.register_webhook(
        url="https://your-app.com/webhook",
        events=["idea.created", "idea.selected"]
    )
    
    print(f"Webhook registered: {webhook.id}")

if __name__ == "__main__":
    asyncio.run(main())
```

```javascript
// examples/javascript_client.js
// Example JavaScript client for TubeSensei API

import { TubeSenseiClient } from '@tubesensei/sdk';

const client = new TubeSenseiClient({
    apiKey: 'your_api_key_here',
    baseURL: 'https://api.tubesensei.com'
});

async function main() {
    try {
        // List channels
        const channels = await client.channels.list({
            limit: 10,
            status: 'active'
        });
        
        console.log(`Found ${channels.pagination.total} channels`);
        
        // Search globally
        const searchResults = await client.search.global({
            q: 'artificial intelligence',
            types: ['ideas', 'videos']
        });
        
        console.log('Search results:', searchResults);
        
        // Get ideas with high confidence
        const ideas = await client.ideas.list({
            minConfidence: 0.8,
            limit: 50
        });
        
        // Export to Excel
        const exportData = await client.export.ideas({
            ideaIds: ideas.data.map(i => i.id),
            format: 'excel'
        });
        
        // Save to file
        const fs = require('fs');
        fs.writeFileSync('ideas.xlsx', exportData);
        
        // Setup webhook
        const webhook = await client.webhooks.create({
            url: 'https://your-app.com/webhook',
            events: ['idea.created'],
            secret: 'your_webhook_secret'
        });
        
        console.log('Webhook created:', webhook.id);
        
    } catch (error) {
        console.error('API Error:', error.response?.data || error.message);
    }
}

main();
```

### 5.4 Integration Testing

```python
# tests/integration/test_complete_flow.py
"""End-to-end integration tests"""

import pytest
import asyncio
from typing import List
from uuid import UUID

from app.tests.fixtures import *
from app.services.channel_service import ChannelService
from app.services.video_service import VideoService
from app.services.idea_service import IdeaService
from app.services.export_service import ExportService

@pytest.mark.asyncio
async def test_complete_workflow(db_session, api_client):
    """Test complete workflow from channel to export"""
    
    # 1. Add a channel
    channel_service = ChannelService(db_session)
    channel = await channel_service.add_channel({
        "youtube_channel_id": "UCtest123",
        "processing_config": {"min_duration": 300}
    })
    
    assert channel.id is not None
    
    # 2. Discover videos (mocked)
    video_service = VideoService(db_session)
    videos = await video_service.discover_channel_videos(channel.id)
    
    assert len(videos) > 0
    
    # 3. Process a video
    video = videos[0]
    job = await video_service.process_video(video.id)
    
    # Wait for processing (mocked to complete immediately in tests)
    await asyncio.sleep(0.1)
    
    # 4. Check ideas were extracted
    idea_service = IdeaService(db_session)
    ideas = await idea_service.list_ideas(video_id=video.id)
    
    assert len(ideas["items"]) > 0
    
    # 5. Select some ideas
    selected_ideas = []
    for idea in ideas["items"][:3]:
        updated = await idea_service.update_idea(
            idea["id"],
            {"status": "selected"}
        )
        selected_ideas.append(updated.id)
    
    # 6. Export selected ideas
    export_service = ExportService(db_session)
    
    # Test JSON export
    json_export = await export_service.export_ideas(
        idea_ids=selected_ideas,
        format="json"
    )
    
    assert json_export.getvalue() is not None
    
    # Test Excel export
    excel_export = await export_service.export_ideas(
        idea_ids=selected_ideas,
        format="excel"
    )
    
    assert excel_export.getvalue() is not None
    
    # 7. Test API access
    response = await api_client.get(
        "/api/v1/channels",
        headers={"X-API-Key": "test_key"}
    )
    
    assert response.status_code == 200
    assert response.json()["success"] is True

@pytest.mark.asyncio
async def test_webhook_delivery(db_session, webhook_service):
    """Test webhook delivery system"""
    
    # Register webhook
    webhook = await webhook_service.register_webhook(
        url="https://httpbin.org/post",  # Test endpoint
        events=["idea.created"],
        secret="test_secret"
    )
    
    # Trigger event
    await webhook_service.trigger_event(
        event_type="idea.created",
        payload={
            "idea_id": "test_123",
            "title": "Test Idea",
            "confidence": 0.85
        }
    )
    
    # Check delivery stats
    stats = await webhook_service.get_webhook_stats(webhook.id)
    
    assert stats["total_deliveries"] > 0
    assert stats["successful_deliveries"] > 0
```

---

## Implementation Checklist

### Day 4 Tasks
- [ ] Implement comprehensive ExportService
- [ ] Create JSON export with options
- [ ] Create CSV export functionality
- [ ] Create Excel export with formatting
- [ ] Implement IdeaHunter format export
- [ ] Create Markdown documentation export
- [ ] Implement ZIP archive export
- [ ] Build export API endpoints
- [ ] Create webhook service
- [ ] Test all export formats

### Day 5 Tasks
- [ ] Enhance OpenAPI documentation
- [ ] Add comprehensive API examples
- [ ] Generate client SDKs
- [ ] Create usage examples in multiple languages
- [ ] Write integration guides
- [ ] Perform end-to-end testing
- [ ] Load test API endpoints
- [ ] Create API documentation site
- [ ] Prepare deployment scripts
- [ ] Final security audit

---

## Testing Requirements

### Export Testing

```python
# tests/test_export.py
import pytest
import json
from io import BytesIO

@pytest.mark.asyncio
async def test_json_export(export_service, sample_ideas):
    """Test JSON export format"""
    buffer = await export_service.export_ideas(
        idea_ids=[i.id for i in sample_ideas],
        format="json",
        options={"include_context": True}
    )
    
    data = json.loads(buffer.getvalue())
    
    assert data["export_version"] == "1.0"
    assert len(data["ideas"]) == len(sample_ideas)
    assert "source" in data["ideas"][0]

@pytest.mark.asyncio
async def test_excel_export(export_service, sample_ideas):
    """Test Excel export format"""
    buffer = await export_service.export_ideas(
        idea_ids=[i.id for i in sample_ideas],
        format="excel"
    )
    
    # Verify it's a valid Excel file
    from openpyxl import load_workbook
    wb = load_workbook(buffer)
    
    assert "Ideas" in wb.sheetnames
    assert "Summary" in wb.sheetnames
    
    ideas_sheet = wb["Ideas"]
    assert ideas_sheet.max_row == len(sample_ideas) + 1  # +1 for header

@pytest.mark.asyncio
async def test_ideahunter_compatibility(export_service, sample_ideas):
    """Test IdeaHunter format compatibility"""
    buffer = await export_service.export_ideas(
        idea_ids=[i.id for i in sample_ideas],
        format="ideahunter"
    )
    
    data = json.loads(buffer.getvalue())
    
    assert data["source"] == "TubeSensei"
    assert "external_id" in data["ideas"][0]
    assert "research_notes" in data["ideas"][0]
    assert "priority" in data["ideas"][0]
```

### Performance Testing

```python
# tests/performance/test_api_load.py
import pytest
import asyncio
from locust import HttpUser, task, between

class TubeSenseiAPIUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Set up API key"""
        self.client.headers.update({
            "X-API-Key": "load_test_key"
        })
    
    @task(3)
    def list_ideas(self):
        """Most common operation"""
        self.client.get("/api/v1/ideas?limit=100")
    
    @task(2)
    def search_ideas(self):
        """Search operation"""
        self.client.get("/api/v1/search?q=technology")
    
    @task(1)
    def export_ideas(self):
        """Export operation"""
        self.client.post("/api/v1/export/ideas", json={
            "idea_ids": ["id1", "id2", "id3"],
            "format": "json"
        })

# Run with: locust -f test_api_load.py --host=http://localhost:8000
```

---

## Success Criteria

### Day 4 Completion
- All export formats implemented and tested
- IdeaHunter integration confirmed
- Webhook system operational
- Export API endpoints functional
- Bulk export working
- ZIP archive includes all formats

### Day 5 Completion
- OpenAPI documentation complete
- SDKs generated for 3+ languages
- Client examples working
- Integration tests passing
- Load tests show acceptable performance
- Documentation website deployed

### Performance Metrics
- Export generation < 5 seconds for 1000 ideas
- Webhook delivery < 1 second
- API documentation loads < 2 seconds
- SDK generation automated
- 95% test coverage achieved

### Quality Standards
- All exports validate in target applications
- Webhooks retry on failure
- API documentation is comprehensive
- SDKs include type definitions
- Examples cover common use cases