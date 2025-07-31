from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Tuple
import os
import torch
import numpy as np
from PIL import Image
import trimesh
import tempfile
import base64
import io
import logging
import asyncio
from datetime import datetime
import json
import time
from pathlib import Path
import uvicorn
import aiofiles
import threading
import queue
import gc
import math
import random
import re
import requests
from scipy.spatial.distance import cdist
from scipy.interpolate import splprep, splev

# Enhanced imports with better error handling
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure optimized logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Enhanced Pydantic Models
class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000, description="Detailed description of the 3D model")
    format: str = Field(default="glb", pattern="^(glb|obj|stl|ply)$", description="Output format: glb, obj, stl, or ply")
    quality: str = Field(default="high", pattern="^(low|medium|high|ultra)$", description="Generation quality level")
    style: str = Field(default="realistic", pattern="^(realistic|stylized|lowpoly|artistic|geometric)$", description="Art style")

class GenerationResponse(BaseModel):
    success: bool
    enhanced_prompt: str
    generation_time: float
    vertices: int
    faces: int
    file_url: str
    preview_image: Optional[str] = None
    format: str
    quality_score: float = 0.0
    complexity_score: float = 0.0
    accuracy_score: float = 0.0
    style_applied: str
    generation_method: str

class StatusResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_name: str
    total_generations: int
    last_generation: Optional[Dict[str, Any]] = None
    version: str
    uptime: str
    features: List[str]
    performance_metrics: Dict[str, Any] = {}
    ai_capabilities: Dict[str, Any] = {}
    
    model_config = {"protected_namespaces": ()}

# FastAPI App with enhanced settings
app = FastAPI(
    title="Advanced AI 3D Asset Generator",
    description="Professional-Grade AI-Powered 3D Asset Creation with High Accuracy",
    version="4.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enhanced CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class AdvancedAI3DAssetGenerator:
    def __init__(self):
        self.model_loaded = False
        self.model_name = "advanced-ai-asset-generator-v4"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generation_history = []
        self.app_start_time = datetime.now()
        self.generation_cache = {}
        self.asset_database = {}
        
        # Enhanced performance metrics
        self.performance_metrics = {
            "avg_generation_time": 0.0,
            "total_cache_hits": 0,
            "models_generated": 0,
            "fastest_generation": float('inf'),
            "slowest_generation": 0.0,
            "accuracy_improvements": 0,
            "complex_assets_generated": 0
        }
        
        # Initialize enhanced LLM
        self._initialize_enhanced_llm()
        
        # Load asset templates and patterns
        self._load_asset_templates()
        
        # Initialize AI-powered analysis
        self._initialize_ai_analysis()
        
        logger.info("✅ Advanced 3D Asset Generator initialized")
    
    def _initialize_enhanced_llm(self):
        """Initialize enhanced LLM with specialized 3D prompts"""
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            try:
                self.llm = ChatGroq(
                    api_key=groq_api_key,
                    model="llama3-70b-8192",  # More powerful model
                    temperature=0.3,
                    max_tokens=500,
                    timeout=15
                )
                
                # Specialized prompt template for 3D asset generation
                self.prompt_template = PromptTemplate(
                    input_variables=["user_prompt", "style", "quality"],
                    template="""You are an expert 3D artist and technical designer. Analyze this request for 3D asset generation:

User Request: "{user_prompt}"
Style: {style}
Quality Level: {quality}

Provide a detailed technical breakdown:

OBJECT_TYPE: [Identify the main object category]
GEOMETRY_BREAKDOWN:
- Primary shapes: [List main geometric components]
- Secondary details: [Important features and details]
- Proportions: [Key measurements and ratios]
- Complexity level: [Simple/Medium/Complex/Highly Complex]

VISUAL_CHARACTERISTICS:
- Surface details: [Textures, patterns, surface features]
- Material properties: [Material types and properties]
- Color scheme: [Primary and secondary colors]
- Lighting considerations: [How light should interact]

CONSTRUCTION_APPROACH:
- Base geometry: [Starting primitive or approach]
- Modification steps: [How to transform base shape]
- Detail addition: [How to add fine details]
- Assembly method: [How parts connect]

TECHNICAL_SPECIFICATIONS:
- Recommended polygon count: [Target vertex/face count]
- Subdivision requirements: [Areas needing more detail]
- Optimization notes: [Performance considerations]

Be extremely specific and technical. Focus on geometric accuracy and realistic proportions."""
                )
                
                self.chain = self.prompt_template | self.llm | StrOutputParser()
                logger.info("✅ Enhanced LLM with specialized 3D prompts initialized")
                
            except Exception as e:
                logger.warning(f"⚠️ Enhanced LLM initialization failed: {e}")
                self.llm = None
                self.chain = None
        else:
            logger.warning("⚠️ GROQ_API_KEY not found")
            self.llm = None
            self.chain = None
    
    def _load_asset_templates(self):
        """Load comprehensive asset templates and patterns"""
        logger.info("🔄 Loading advanced asset templates...")
        
        self.asset_templates = {
            # Vehicles
            'vehicles': {
                'car': {
                    'base_proportions': {'length': 4.5, 'width': 1.8, 'height': 1.5},
                    'components': ['body', 'wheels', 'windows', 'lights', 'doors'],
                    'detail_level': {'low': 200, 'medium': 800, 'high': 2000, 'ultra': 5000}
                },
                'truck': {
                    'base_proportions': {'length': 6.0, 'width': 2.2, 'height': 2.5},
                    'components': ['cab', 'bed', 'wheels', 'windows', 'grille'],
                    'detail_level': {'low': 300, 'medium': 1000, 'high': 2500, 'ultra': 6000}
                },
                'motorcycle': {
                    'base_proportions': {'length': 2.2, 'width': 0.8, 'height': 1.2},
                    'components': ['frame', 'wheels', 'engine', 'handlebars', 'seat'],
                    'detail_level': {'low': 150, 'medium': 600, 'high': 1500, 'ultra': 3500}
                }
            },
            
            # Architecture
            'buildings': {
                'house': {
                    'base_proportions': {'length': 10, 'width': 8, 'height': 6},
                    'components': ['walls', 'roof', 'windows', 'doors', 'foundation'],
                    'detail_level': {'low': 100, 'medium': 500, 'high': 1200, 'ultra': 3000}
                },
                'skyscraper': {
                    'base_proportions': {'length': 20, 'width': 20, 'height': 100},
                    'components': ['base', 'floors', 'windows', 'roof', 'details'],
                    'detail_level': {'low': 200, 'medium': 1000, 'high': 3000, 'ultra': 8000}
                }
            },
            
            # Natural objects
            'nature': {
                'tree': {
                    'base_proportions': {'trunk_height': 3, 'crown_radius': 2, 'trunk_radius': 0.3},
                    'components': ['trunk', 'branches', 'leaves', 'roots'],
                    'detail_level': {'low': 100, 'medium': 800, 'high': 2500, 'ultra': 8000}
                },
                'flower': {
                    'base_proportions': {'stem_height': 1.5, 'flower_radius': 0.3},
                    'components': ['stem', 'petals', 'center', 'leaves'],
                    'detail_level': {'low': 50, 'medium': 300, 'high': 1000, 'ultra': 3000}
                }
            },
            
            # Furniture
            'furniture': {
                'chair': {
                    'base_proportions': {'width': 0.5, 'depth': 0.5, 'height': 0.9},
                    'components': ['seat', 'backrest', 'legs', 'armrests'],
                    'detail_level': {'low': 50, 'medium': 200, 'high': 800, 'ultra': 2000}
                },
                'table': {
                    'base_proportions': {'width': 1.5, 'depth': 0.8, 'height': 0.75},
                    'components': ['top', 'legs', 'supports', 'details'],
                    'detail_level': {'low': 30, 'medium': 150, 'high': 600, 'ultra': 1500}
                }
            },
            
            # Characters and creatures
            'characters': {
                'human': {
                    'base_proportions': {'height': 1.75, 'head_size': 0.2, 'shoulder_width': 0.5},
                    'components': ['head', 'torso', 'arms', 'legs', 'hands', 'feet'],
                    'detail_level': {'low': 500, 'medium': 2000, 'high': 8000, 'ultra': 20000}
                },
                'animal': {
                    'base_proportions': {'length': 1.0, 'height': 0.6, 'width': 0.3},
                    'components': ['body', 'head', 'legs', 'tail', 'features'],
                    'detail_level': {'low': 300, 'medium': 1200, 'high': 4000, 'ultra': 12000}
                }
            },
            
            # Weapons and tools
            'weapons': {
                'sword': {
                    'base_proportions': {'length': 1.0, 'blade_width': 0.05, 'handle_length': 0.2},
                    'components': ['blade', 'guard', 'handle', 'pommel', 'details'],
                    'detail_level': {'low': 100, 'medium': 400, 'high': 1200, 'ultra': 3000}
                },
                'gun': {
                    'base_proportions': {'length': 0.8, 'height': 0.15, 'width': 0.05},
                    'components': ['barrel', 'stock', 'trigger', 'sights', 'details'],
                    'detail_level': {'low': 150, 'medium': 600, 'high': 1800, 'ultra': 4500}
                }
            }
        }
        
        logger.info(f"✅ Loaded {len(self.asset_templates)} asset categories")
        self.model_loaded = True
    
    def _initialize_ai_analysis(self):
        """Initialize AI-powered object analysis"""
        logger.info("🧠 Initializing AI analysis capabilities...")
        
        # Object classification patterns
        self.classification_patterns = {
            'vehicles': ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'plane', 'helicopter', 'boat', 'ship', 'train'],
            'buildings': ['house', 'building', 'skyscraper', 'church', 'castle', 'tower', 'bridge', 'stadium'],
            'nature': ['tree', 'flower', 'plant', 'mountain', 'rock', 'cloud', 'water', 'grass', 'bush'],
            'furniture': ['chair', 'table', 'sofa', 'bed', 'desk', 'shelf', 'cabinet', 'lamp'],
            'characters': ['person', 'human', 'character', 'robot', 'alien', 'monster', 'animal', 'creature'],
            'weapons': ['sword', 'gun', 'rifle', 'pistol', 'knife', 'axe', 'hammer', 'bow', 'arrow'],
            'electronics': ['phone', 'computer', 'laptop', 'tv', 'radio', 'camera', 'drone'],
            'food': ['apple', 'banana', 'cake', 'pizza', 'burger', 'bottle', 'cup', 'plate'],
            'abstract': ['sculpture', 'art', 'crystal', 'gem', 'spiral', 'wave', 'pattern']
        }
        
        logger.info("✅ AI analysis capabilities initialized")
    
    async def analyze_prompt_with_ai(self, prompt: str, style: str = "realistic", quality: str = "high") -> Dict[str, Any]:
        """Advanced AI-powered prompt analysis"""
        try:
            if not self.chain:
                return self._analyze_prompt_locally(prompt, style, quality)
            
            # Get AI analysis
            analysis = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.chain.invoke({
                        "user_prompt": prompt, 
                        "style": style, 
                        "quality": quality
                    })
                ),
                timeout=10.0
            )
            
            # Parse the AI response
            parsed_analysis = self._parse_ai_analysis(analysis, prompt)
            return parsed_analysis
            
        except Exception as e:
            logger.warning(f"⚠️ AI analysis failed: {e}")
            return self._analyze_prompt_locally(prompt, style, quality)
    
    def _parse_ai_analysis(self, analysis: str, original_prompt: str) -> Dict[str, Any]:
        """Parse AI analysis response into structured data"""
        try:
            result = {
                'object_type': 'unknown',
                'category': 'abstract',
                'complexity': 'medium',
                'components': [],
                'materials': [],
                'colors': [],
                'proportions': {},
                'construction_method': 'procedural',
                'target_vertices': 1000,
                'enhanced_prompt': analysis,
                'confidence': 0.8
            }
            
            analysis_lower = analysis.lower()
            
            # Extract object type
            for category, objects in self.classification_patterns.items():
                for obj in objects:
                    if obj in analysis_lower or obj in original_prompt.lower():
                        result['object_type'] = obj
                        result['category'] = category
                        break
                if result['object_type'] != 'unknown':
                    break
            
            # Extract complexity level
            if 'highly complex' in analysis_lower:
                result['complexity'] = 'ultra'
                result['target_vertices'] = 5000
            elif 'complex' in analysis_lower:
                result['complexity'] = 'high'
                result['target_vertices'] = 2000
            elif 'medium' in analysis_lower:
                result['complexity'] = 'medium'
                result['target_vertices'] = 1000
            elif 'simple' in analysis_lower:
                result['complexity'] = 'low'
                result['target_vertices'] = 500
            
            # Extract colors
            colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white', 'brown', 'gray', 'pink']
            for color in colors:
                if color in analysis_lower:
                    result['colors'].append(color)
            
            # Extract materials
            materials = ['metal', 'wood', 'glass', 'plastic', 'stone', 'fabric', 'ceramic', 'leather', 'rubber']
            for material in materials:
                if material in analysis_lower:
                    result['materials'].append(material)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Analysis parsing failed: {e}")
            return self._analyze_prompt_locally(original_prompt, "realistic", "high")
    
    def _analyze_prompt_locally(self, prompt: str, style: str, quality: str) -> Dict[str, Any]:
        """Enhanced local prompt analysis with better pattern recognition"""
        prompt_lower = prompt.lower()
        
        result = {
            'object_type': 'unknown',
            'category': 'abstract',
            'complexity': quality.lower(),
            'components': [],
            'materials': [],
            'colors': [],
            'proportions': {},
            'construction_method': 'procedural',
            'target_vertices': {'low': 500, 'medium': 1000, 'high': 2000, 'ultra': 5000}[quality.lower()],
            'enhanced_prompt': prompt,
            'confidence': 0.6
        }
        
        # Advanced object classification
        max_matches = 0
        best_category = 'abstract'
        best_object = 'unknown'
        
        for category, objects in self.classification_patterns.items():
            matches = sum(1 for obj in objects if obj in prompt_lower)
            if matches > max_matches:
                max_matches = matches
                best_category = category
                # Find the specific object with most relevance
                for obj in objects:
                    if obj in prompt_lower:
                        best_object = obj
                        break
        
        result['category'] = best_category
        result['object_type'] = best_object
        
        # Enhanced feature extraction
        self._extract_colors(prompt_lower, result)
        self._extract_materials(prompt_lower, result)
        self._extract_complexity_indicators(prompt_lower, result)
        self._extract_proportions(prompt_lower, result)
        
        return result
    
    def _extract_colors(self, prompt: str, result: Dict):
        """Extract color information with advanced pattern matching"""
        color_patterns = {
            'red': ['red', 'crimson', 'scarlet', 'cherry', 'burgundy'],
            'blue': ['blue', 'azure', 'navy', 'cyan', 'cobalt'],
            'green': ['green', 'emerald', 'forest', 'lime', 'olive'],
            'yellow': ['yellow', 'gold', 'amber', 'lemon', 'cream'],
            'purple': ['purple', 'violet', 'lavender', 'plum', 'magenta'],
            'orange': ['orange', 'amber', 'peach', 'coral', 'bronze'],
            'black': ['black', 'dark', 'ebony', 'charcoal'],
            'white': ['white', 'ivory', 'pearl', 'snow', 'cream'],
            'brown': ['brown', 'tan', 'chocolate', 'coffee', 'wood'],
            'gray': ['gray', 'grey', 'silver', 'steel', 'slate']
        }
        
        for base_color, variations in color_patterns.items():
            if any(variant in prompt for variant in variations):
                result['colors'].append(base_color)
    
    def _extract_materials(self, prompt: str, result: Dict):
        """Extract material information with detailed patterns"""
        material_patterns = {
            'metal': ['metal', 'steel', 'iron', 'aluminum', 'copper', 'bronze', 'chrome'],
            'wood': ['wood', 'wooden', 'oak', 'pine', 'mahogany', 'bamboo'],
            'glass': ['glass', 'crystal', 'transparent', 'clear'],
            'plastic': ['plastic', 'polymer', 'synthetic'],
            'stone': ['stone', 'marble', 'granite', 'rock', 'concrete'],
            'fabric': ['fabric', 'cloth', 'textile', 'canvas', 'silk'],
            'ceramic': ['ceramic', 'pottery', 'porcelain'],
            'leather': ['leather', 'hide', 'skin'],
            'rubber': ['rubber', 'elastic', 'flexible']
        }
        
        for material, variations in material_patterns.items():
            if any(variant in prompt for variant in variations):
                result['materials'].append(material)
    
    def _extract_complexity_indicators(self, prompt: str, result: Dict):
        """Extract complexity indicators from prompt"""
        complexity_indicators = {
            'ultra': ['highly detailed', 'extremely complex', 'intricate', 'elaborate', 'sophisticated'],
            'high': ['detailed', 'complex', 'refined', 'advanced'],
            'medium': ['standard', 'normal', 'regular'],
            'low': ['simple', 'basic', 'minimal', 'clean']
        }
        
        for level, indicators in complexity_indicators.items():
            if any(indicator in prompt for indicator in indicators):
                result['complexity'] = level
                result['target_vertices'] = {'low': 500, 'medium': 1000, 'high': 2000, 'ultra': 5000}[level]
                break
    
    def _extract_proportions(self, prompt: str, result: Dict):
        """Extract size and proportion information"""
        size_indicators = {
            'large': ['large', 'big', 'huge', 'massive', 'giant'],
            'small': ['small', 'tiny', 'mini', 'compact'],
            'tall': ['tall', 'high', 'towering'],
            'wide': ['wide', 'broad', 'thick'],
            'long': ['long', 'extended', 'elongated'],
            'short': ['short', 'brief', 'compact']
        }
        
        for size, indicators in size_indicators.items():
            if any(indicator in prompt for indicator in indicators):
                result['proportions'][size] = True
    
    async def generate_advanced_3d_asset(self, prompt: str, format_type: str = "glb", 
                                       quality: str = "high", style: str = "realistic") -> Tuple[trimesh.Trimesh, Dict[str, Any], float]:
        """Generate advanced 3D asset with high accuracy"""
        start_time = time.time()
        
        try:
            # Analyze prompt with AI
            analysis = await self.analyze_prompt_with_ai(prompt, style, quality)
            
            logger.info(f"🔍 Analysis: {analysis['object_type']} ({analysis['category']}) - {analysis['complexity']} complexity")
            
            # Generate mesh based on analysis
            mesh = await self._generate_mesh_from_analysis(analysis, style, quality)
            
            # Apply post-processing
            mesh = self._apply_advanced_post_processing(mesh, analysis, style)
            
            generation_time = time.time() - start_time
            
            # Calculate advanced metrics
            quality_score = self._calculate_quality_score(mesh, analysis)
            complexity_score = self._calculate_complexity_score(mesh, analysis)
            accuracy_score = self._calculate_accuracy_score(mesh, analysis, prompt)
            
            # Update performance metrics
            self._update_performance_metrics(generation_time, quality_score, complexity_score)
            
            # Store in history
            generation_data = {
                'prompt': prompt,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat(),
                'generation_time': generation_time,
                'vertices': len(mesh.vertices),
                'faces': len(mesh.faces),
                'quality_score': quality_score,
                'complexity_score': complexity_score,
                'accuracy_score': accuracy_score,
                'style': style,
                'format': format_type
            }
            
            self.generation_history.append(generation_data)
            
            logger.info(f"✅ Generated {analysis['object_type']} in {generation_time:.2f}s - Q:{quality_score:.1f} C:{complexity_score:.1f} A:{accuracy_score:.1f}")
            
            return mesh, analysis, generation_time
            
        except Exception as e:
            logger.error(f"❌ Advanced generation failed: {e}")
            # Return enhanced fallback
            fallback_mesh = self._create_intelligent_fallback(prompt, quality)
            generation_time = time.time() - start_time
            analysis = {'object_type': 'fallback', 'category': 'abstract', 'enhanced_prompt': prompt}
            return fallback_mesh, analysis, generation_time
    
    async def _generate_mesh_from_analysis(self, analysis: Dict[str, Any], style: str, quality: str) -> trimesh.Trimesh:
        """Generate mesh based on AI analysis"""
        object_type = analysis['object_type']
        category = analysis['category']
        
        # Route to specialized generators
        if category == 'vehicles':
            return await self._generate_vehicle(analysis, style, quality)
        elif category == 'buildings':
            return await self._generate_building(analysis, style, quality)
        elif category == 'nature':
            return await self._generate_natural_object(analysis, style, quality)
        elif category == 'furniture':
            return await self._generate_furniture(analysis, style, quality)
        elif category == 'characters':
            return await self._generate_character(analysis, style, quality)
        elif category == 'weapons':
            return await self._generate_weapon(analysis, style, quality)
        else:
            return await self._generate_procedural_object(analysis, style, quality)
    
    async def _generate_vehicle(self, analysis: Dict[str, Any], style: str, quality: str) -> trimesh.Trimesh:
        """Generate detailed vehicle models"""
        object_type = analysis['object_type']
        
        if object_type == 'car':
            return await self._generate_detailed_car(analysis, style, quality)
        elif object_type == 'truck':
            return await self._generate_detailed_truck(analysis, style, quality)
        elif object_type == 'motorcycle':
            return await self._generate_detailed_motorcycle(analysis, style, quality)
        else:
            return await self._generate_generic_vehicle(analysis, style, quality)
    
    async def _generate_detailed_car(self, analysis: Dict[str, Any], style: str, quality: str) -> trimesh.Trimesh:
        """Generate highly detailed car model"""
        parts = []
        
        # Get quality-based detail level
        detail_levels = {'low': 200, 'medium': 800, 'high': 2000, 'ultra': 5000}
        target_detail = detail_levels.get(quality, 1000)
        
        # Main body with proper proportions
        body_length = 4.5
        body_width = 1.8
        body_height = 1.2
        
        # Create main body using multiple segments for realism
        body_segments = []
        
        # Front section (hood)
        front_body = trimesh.creation.box(extents=[body_length * 0.3, body_width, body_height * 0.8])
        front_body.apply_translation([body_length * 0.25, 0, body_height * 0.4])
        body_segments.append(front_body)
        
        # Middle section (passenger area)
        mid_body = trimesh.creation.box(extents=[body_length * 0.4, body_width, body_height])
        mid_body.apply_translation([0, 0, body_height * 0.5])
        body_segments.append(mid_body)
        
        # Rear section (trunk)
        rear_body = trimesh.creation.box(extents=[body_length * 0.3, body_width, body_height * 0.9])
        rear_body.apply_translation([-body_length * 0.25, 0, body_height * 0.45])
        body_segments.append(rear_body)
        
        # Windshield and windows
        windshield = trimesh.creation.box(extents=[0.05, body_width * 0.9, body_height * 0.6])
        windshield.apply_translation([body_length * 0.15, 0, body_height * 0.8])
        windshield.apply_transform(trimesh.transformations.rotation_matrix(np.radians(-15), [0, 1, 0]))
        body_segments.append(windshield)
        
        # Side windows
        for side in [-1, 1]:
            window = trimesh.creation.box(extents=[body_length * 0.35, 0.05, body_height * 0.4])
            window.apply_translation([0, side * body_width * 0.48, body_height * 0.75])
            body_segments.append(window)
        
        # Wheels with realistic proportions
        wheel_radius = 0.35
        wheel_width = 0.25
        wheel_positions = [
            [body_length * 0.3, -body_width * 0.55, wheel_radius],
            [body_length * 0.3, body_width * 0.55, wheel_radius],
            [-body_length * 0.3, -body_width * 0.55, wheel_radius],
            [-body_length * 0.3, body_width * 0.55, wheel_radius]
        ]
        
        for pos in wheel_positions:
            # Wheel rim
            wheel = trimesh.creation.cylinder(radius=wheel_radius, height=wheel_width, sections=16)
            wheel.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
            wheel.apply_translation(pos)
            
            # Tire tread pattern
            if quality in ['high', 'ultra']:
                tread_count = 12 if quality == 'ultra' else 8
                for i in range(tread_count):
                    angle = i * 2 * np.pi / tread_count
                    tread = trimesh.creation.box(extents=[0.02, wheel_width * 0.8, 0.15])
                    tread_pos = [
                        pos[0] + wheel_radius * 0.9 * np.cos(angle),
                        pos[1],
                        pos[2] + wheel_radius * 0.9 * np.sin(angle)
                    ]
                    tread.apply_translation(tread_pos)
                    body_segments.append(tread)
            
            body_segments.append(wheel)
        
        # Detailed features based on quality
        if quality in ['high', 'ultra']:
            # Headlights
            for side in [-1, 1]:
                headlight = trimesh.creation.uv_sphere(radius=0.15, count=[8, 8])
                headlight.apply_translation([body_length * 0.42, side * body_width * 0.3, body_height * 0.6])
                body_segments.append(headlight)
            
            # Grille
            grille = trimesh.creation.box(extents=[0.05, body_width * 0.6, body_height * 0.3])
            grille.apply_translation([body_length * 0.45, 0, body_height * 0.5])
            body_segments.append(grille)
            
            # Door handles
            for side in [-1, 1]:
                for door_pos in [0.1, -0.1]:
                    handle = trimesh.creation.box(extents=[0.15, 0.03, 0.05])
                    handle.apply_translation([door_pos, side * body_width * 0.52, body_height * 0.6])
                    body_segments.append(handle)
        
        if quality == 'ultra':
            # Side mirrors
            for side in [-1, 1]:
                mirror_arm = trimesh.creation.cylinder(radius=0.02, height=0.15, sections=6)
                mirror_arm.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, 1]))
                mirror_arm.apply_translation([body_length * 0.1, side * body_width * 0.6, body_height * 0.9])
                
                mirror_glass = trimesh.creation.box(extents=[0.08, 0.05, 0.12])
                mirror_glass.apply_translation([body_length * 0.1, side * body_width * 0.65, body_height * 0.9])
                
                body_segments.extend([mirror_arm, mirror_glass])
            
            # Exhaust pipe
            exhaust = trimesh.creation.cylinder(radius=0.04, height=0.3, sections=8)
            exhaust.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
            exhaust.apply_translation([-body_length * 0.45, -body_width * 0.3, 0.1])
            body_segments.append(exhaust)
        
        # Combine all parts
        car_mesh = trimesh.util.concatenate(body_segments)
        
        # Apply colors based on analysis
        colors = analysis.get('colors', ['gray'])
        primary_color = colors[0] if colors else 'gray'
        car_mesh = self._apply_vehicle_colors(car_mesh, primary_color, style)
        
        return car_mesh
    
    async def _generate_detailed_truck(self, analysis: Dict[str, Any], style: str, quality: str) -> trimesh.Trimesh:
        """Generate detailed truck model"""
        parts = []
        
        # Truck proportions
        cab_length = 2.5
        cab_width = 2.2
        cab_height = 2.5
        bed_length = 3.5
        bed_height = 1.5
        
        # Cab
        cab = trimesh.creation.box(extents=[cab_length, cab_width, cab_height])
        cab.apply_translation([cab_length/2, 0, cab_height/2])
        parts.append(cab)
        
        # Bed
        bed_base = trimesh.creation.box(extents=[bed_length, cab_width, 0.2])
        bed_base.apply_translation([-bed_length/2 - 0.5, 0, bed_height/2])
        parts.append(bed_base)
        
        # Bed sides
        for side in [-1, 1]:
            bed_side = trimesh.creation.box(extents=[bed_length, 0.1, bed_height])
            bed_side.apply_translation([-bed_length/2 - 0.5, side * cab_width/2, bed_height/2])
            parts.append(bed_side)
        
        # Tailgate
        tailgate = trimesh.creation.box(extents=[0.1, cab_width, bed_height])
        tailgate.apply_translation([-bed_length - 0.5, 0, bed_height/2])
        parts.append(tailgate)
        
        # Wheels
        wheel_radius = 0.5
        wheel_positions = [
            [cab_length * 0.3, -cab_width * 0.6, wheel_radius],
            [cab_length * 0.3, cab_width * 0.6, wheel_radius],
            [-bed_length * 0.3, -cab_width * 0.6, wheel_radius],
            [-bed_length * 0.3, cab_width * 0.6, wheel_radius]
        ]
        
        for pos in wheel_positions:
            wheel = trimesh.creation.cylinder(radius=wheel_radius, height=0.3, sections=12)
            wheel.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
            wheel.apply_translation(pos)
            parts.append(wheel)
        
        # Combine parts
        truck_mesh = trimesh.util.concatenate(parts)
        
        # Apply colors
        colors = analysis.get('colors', ['blue'])
        primary_color = colors[0] if colors else 'blue'
        truck_mesh = self._apply_vehicle_colors(truck_mesh, primary_color, style)
        
        return truck_mesh
    
    async def _generate_detailed_motorcycle(self, analysis: Dict[str, Any], style: str, quality: str) -> trimesh.Trimesh:
        """Generate detailed motorcycle model"""
        parts = []
        
        # Frame
        frame_main = trimesh.creation.cylinder(radius=0.03, height=1.5, sections=8)
        frame_main.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
        frame_main.apply_translation([0, 0, 0.6])
        parts.append(frame_main)
        
        # Engine
        engine = trimesh.creation.box(extents=[0.4, 0.3, 0.3])
        engine.apply_translation([0, 0, 0.4])
        parts.append(engine)
        
        # Wheels
        front_wheel = trimesh.creation.cylinder(radius=0.35, height=0.15, sections=16)
        front_wheel.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
        front_wheel.apply_translation([1.0, 0, 0.35])
        
        rear_wheel = trimesh.creation.cylinder(radius=0.35, height=0.15, sections=16)
        rear_wheel.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
        rear_wheel.apply_translation([-1.0, 0, 0.35])
        
        parts.extend([front_wheel, rear_wheel])
        
        # Handlebars
        handlebar = trimesh.creation.cylinder(radius=0.02, height=0.6, sections=6)
        handlebar.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 0, 1]))
        handlebar.apply_translation([0.8, 0, 1.0])
        parts.append(handlebar)
        
        # Seat
        seat = trimesh.creation.box(extents=[0.4, 0.2, 0.1])
        seat.apply_translation([-0.2, 0, 0.8])
        parts.append(seat)
        
        # Combine parts
        motorcycle_mesh = trimesh.util.concatenate(parts)
        
        # Apply colors
        colors = analysis.get('colors', ['black'])
        primary_color = colors[0] if colors else 'black'
        motorcycle_mesh = self._apply_vehicle_colors(motorcycle_mesh, primary_color, style)
        
        return motorcycle_mesh
    
    async def _generate_building(self, analysis: Dict[str, Any], style: str, quality: str) -> trimesh.Trimesh:
        """Generate detailed building models"""
        object_type = analysis['object_type']
        
        if object_type == 'house':
            return await self._generate_detailed_house(analysis, style, quality)
        elif object_type == 'skyscraper':
            return await self._generate_detailed_skyscraper(analysis, style, quality)
        else:
            return await self._generate_generic_building(analysis, style, quality)
    
    async def _generate_detailed_house(self, analysis: Dict[str, Any], style: str, quality: str) -> trimesh.Trimesh:
        """Generate detailed house model"""
        parts = []
        
        # Base structure
        house_width = 8
        house_depth = 6
        house_height = 3
        
        # Main walls
        main_structure = trimesh.creation.box(extents=[house_width, house_depth, house_height])
        main_structure.apply_translation([0, 0, house_height/2])
        parts.append(main_structure)
        
        # Roof
        roof_vertices = np.array([
            [-house_width/2, -house_depth/2, house_height],
            [house_width/2, -house_depth/2, house_height],
            [house_width/2, house_depth/2, house_height],
            [-house_width/2, house_depth/2, house_height],
            [0, -house_depth/2, house_height + 2],
            [0, house_depth/2, house_height + 2]
        ])
        
        roof_faces = np.array([
            [0, 1, 4], [1, 2, 5], [2, 3, 5], [3, 0, 4],
            [4, 5, 1], [4, 1, 0], [5, 4, 3], [5, 3, 2]
        ])
        
        roof = trimesh.Trimesh(vertices=roof_vertices, faces=roof_faces)
        parts.append(roof)
        
        # Windows
        if quality in ['medium', 'high', 'ultra']:
            window_size = 1.2
            window_positions = [
                [-house_width/3, -house_depth/2 - 0.05, house_height/2],
                [house_width/3, -house_depth/2 - 0.05, house_height/2],
                [-house_width/3, house_depth/2 + 0.05, house_height/2],
                [house_width/3, house_depth/2 + 0.05, house_height/2]
            ]
            
            for pos in window_positions:
                window = trimesh.creation.box(extents=[window_size, 0.1, window_size])
                window.apply_translation(pos)
                parts.append(window)
        
        # Door
        door = trimesh.creation.box(extents=[0.8, 0.1, 2.0])
        door.apply_translation([0, -house_depth/2 - 0.05, 1.0])
        parts.append(door)
        
        # Chimney
        if quality in ['high', 'ultra']:
            chimney = trimesh.creation.box(extents=[0.6, 0.6, 1.5])
            chimney.apply_translation([house_width/3, 0, house_height + 1.75])
            parts.append(chimney)
        
        # Combine parts
        house_mesh = trimesh.util.concatenate(parts)
        
        # Apply materials
        house_mesh = self._apply_building_materials(house_mesh, analysis, style)
        
        return house_mesh
    
    async def _generate_natural_object(self, analysis: Dict[str, Any], style: str, quality: str) -> trimesh.Trimesh:
        """Generate natural objects like trees, flowers, etc."""
        object_type = analysis['object_type']
        
        if object_type == 'tree':
            return await self._generate_advanced_tree(analysis, style, quality)
        elif object_type == 'flower':
            return await self._generate_advanced_flower(analysis, style, quality)
        else:
            return await self._generate_generic_natural_object(analysis, style, quality)
    
    async def _generate_advanced_tree(self, analysis: Dict[str, Any], style: str, quality: str) -> trimesh.Trimesh:
        """Generate advanced tree with realistic branching"""
        parts = []
        
        # Trunk with realistic tapering
        trunk_height = 4.0
        trunk_base_radius = 0.3
        trunk_top_radius = 0.15
        
        # Create tapered trunk using multiple cylinders
        trunk_segments = 8 if quality in ['high', 'ultra'] else 4
        
        for i in range(trunk_segments):
            segment_height = trunk_height / trunk_segments
            segment_y = i * segment_height
            
            # Calculate radius at this height
            t = i / (trunk_segments - 1) if trunk_segments > 1 else 0
            radius = trunk_base_radius * (1 - t) + trunk_top_radius * t
            
            segment = trimesh.creation.cylinder(radius=radius, height=segment_height, sections=12)
            segment.apply_translation([0, 0, segment_y + segment_height/2])
            parts.append(segment)
        
        # Advanced branching system
        if quality in ['medium', 'high', 'ultra']:
            branch_count = {'medium': 4, 'high': 8, 'ultra': 12}[quality]
            
            for i in range(branch_count):
                angle = i * 2 * np.pi / branch_count
                branch_height = trunk_height * (0.6 + 0.2 * random.random())
                
                # Main branch
                branch_length = 1.5 + random.random()
                branch_radius = 0.05 + random.random() * 0.03
                
                branch = trimesh.creation.cylinder(radius=branch_radius, height=branch_length, sections=6)
                
                # Branch direction
                branch_angle = np.radians(30 + random.random() * 30)  # 30-60 degrees up
                branch.apply_transform(trimesh.transformations.rotation_matrix(branch_angle, [1, 0, 0]))
                branch.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
                
                branch_pos = [0, 0, branch_height]
                branch.apply_translation(branch_pos)
                parts.append(branch)
                
                # Sub-branches for higher quality
                if quality == 'ultra':
                    for j in range(2):
                        sub_branch = trimesh.creation.cylinder(radius=branch_radius*0.6, height=branch_length*0.7, sections=6)
                        sub_angle = np.radians(20 + random.random() * 20)
                        sub_branch.apply_transform(trimesh.transformations.rotation_matrix(sub_angle, [1, 0, 0]))
                        sub_branch.apply_transform(trimesh.transformations.rotation_matrix(angle + j*np.pi/3, [0, 0, 1]))
                        
                        sub_pos = [
                            branch_length * 0.7 * np.cos(angle) * np.cos(branch_angle),
                            branch_length * 0.7 * np.sin(angle) * np.cos(branch_angle),
                            branch_height + branch_length * 0.7 * np.sin(branch_angle)
                        ]
                        sub_branch.apply_translation(sub_pos)
                        parts.append(sub_branch)
        
        # Foliage/Crown
        crown_positions = [
            [0, 0, trunk_height + 0.5],
            [-0.8, 0.8, trunk_height + 0.2],
            [0.8, -0.8, trunk_height + 0.2],
            [0.8, 0.8, trunk_height + 0.3],
            [-0.8, -0.8, trunk_height + 0.3]
        ]
        
        for pos in crown_positions:
            foliage_size = 0.8 + random.random() * 0.4
            foliage = trimesh.creation.icosphere(subdivisions=2, radius=foliage_size)
            foliage.apply_translation(pos)
            parts.append(foliage)
        
        # Combine all parts
        tree_mesh = trimesh.util.concatenate(parts)
        
        # Apply natural colors
        tree_mesh = self._apply_natural_colors(tree_mesh, 'tree', analysis)
        
        return tree_mesh
    
    def _apply_vehicle_colors(self, mesh: trimesh.Trimesh, primary_color: str, style: str) -> trimesh.Trimesh:
        """Apply realistic vehicle colors"""
        color_map = {
            'red': [180, 40, 40, 255],
            'blue': [40, 80, 180, 255],
            'green': [40, 120, 40, 255],
            'black': [30, 30, 30, 255],
            'white': [220, 220, 220, 255],
            'gray': [100, 100, 100, 255],
            'silver': [160, 160, 160, 255]
        }
        
        base_color = color_map.get(primary_color, color_map['gray'])
        
        # Apply color with slight variation for realism
        colors = np.array([base_color] * len(mesh.vertices), dtype=np.uint8)
        
        if style == 'realistic':
            # Add subtle metallic variation
            variation = np.random.randint(-15, 15, (len(mesh.vertices), 3))
            colors[:, :3] = np.clip(colors[:, :3] + variation, 20, 255)
        
        mesh.visual.vertex_colors = colors
        return mesh
    
    def _apply_building_materials(self, mesh: trimesh.Trimesh, analysis: Dict[str, Any], style: str) -> trimesh.Trimesh:
        """Apply building materials and colors"""
        materials = analysis.get('materials', ['concrete'])
        colors = analysis.get('colors', ['gray'])
        
        # Material-based coloring
        if 'brick' in materials or 'red' in colors:
            base_color = [150, 80, 60, 255]
        elif 'wood' in materials or 'brown' in colors:
            base_color = [120, 80, 40, 255]
        elif 'stone' in materials:
            base_color = [140, 140, 120, 255]
        else:
            base_color = [120, 120, 120, 255]
        
        colors = np.array([base_color] * len(mesh.vertices), dtype=np.uint8)
        
        # Add texture variation
        variation = np.random.randint(-20, 20, (len(mesh.vertices), 3))
        colors[:, :3] = np.clip(colors[:, :3] + variation, 0, 255)
        
        mesh.visual.vertex_colors = colors
        return mesh
    
    def _apply_natural_colors(self, mesh: trimesh.Trimesh, object_type: str, analysis: Dict[str, Any]) -> trimesh.Trimesh:
        """Apply natural colors to organic objects"""
        if object_type == 'tree':
            # Brown trunk, green foliage
            vertex_count = len(mesh.vertices)
            colors = np.zeros((vertex_count, 4), dtype=np.uint8)
            
            for i, vertex in enumerate(mesh.vertices):
                if vertex[2] < 2.0:  # Trunk area
                    colors[i] = [101, 67, 33, 255]  # Brown
                else:  # Foliage area
                    colors[i] = [34, 139, 34, 255]  # Forest green
        else:
            # Default natural green
            colors = np.array([[60, 140, 60, 255]] * len(mesh.vertices), dtype=np.uint8)
        
        # Add natural variation
        variation = np.random.randint(-25, 25, (len(mesh.vertices), 3))
        colors[:, :3] = np.clip(colors[:, :3] + variation, 0, 255)
        
        mesh.visual.vertex_colors = colors
        return mesh
    
    async def _generate_furniture(self, analysis: Dict[str, Any], style: str, quality: str) -> trimesh.Trimesh:
        """Generate furniture objects"""
        object_type = analysis['object_type']
        
        if object_type == 'chair':
            return await self._generate_detailed_chair(analysis, style, quality)
        elif object_type == 'table':
            return await self._generate_detailed_table(analysis, style, quality)
        else:
            return await self._generate_generic_furniture(analysis, style, quality)
    
    async def _generate_detailed_chair(self, analysis: Dict[str, Any], style: str, quality: str) -> trimesh.Trimesh:
        """Generate detailed chair model"""
        parts = []
        
        # Legs
        leg_height = 0.45
        leg_thickness = 0.05
        leg_positions = [
            [-0.2, -0.2, leg_height/2],
            [0.2, -0.2, leg_height/2],
            [-0.2, 0.2, leg_height/2],
            [0.2, 0.2, leg_height/2]
        ]
        
        for pos in leg_positions:
            leg = trimesh.creation.cylinder(radius=leg_thickness/2, height=leg_height, sections=8)
            leg.apply_translation(pos)
            parts.append(leg)
        
        # Seat
        seat = trimesh.creation.box(extents=[0.5, 0.5, 0.05])
        seat.apply_translation([0, 0, leg_height + 0.025])
        parts.append(seat)
        
        # Backrest
        backrest = trimesh.creation.box(extents=[0.5, 0.05, 0.4])
        backrest.apply_translation([0, -0.225, leg_height + 0.25])
        parts.append(backrest)
        
        # Combine parts
        chair_mesh = trimesh.util.concatenate(parts)
        
        # Apply wood material
        chair_mesh = self._apply_furniture_materials(chair_mesh, analysis)
        
        return chair_mesh
    
    def _apply_furniture_materials(self, mesh: trimesh.Trimesh, analysis: Dict[str, Any]) -> trimesh.Trimesh:
        """Apply furniture materials"""
        materials = analysis.get('materials', ['wood'])
        
        if 'wood' in materials:
            base_color = [139, 90, 43, 255]  # Wood brown
        elif 'metal' in materials:
            base_color = [169, 169, 169, 255]  # Metal gray
        else:
            base_color = [160, 120, 80, 255]  # Default furniture color
        
        colors = np.array([base_color] * len(mesh.vertices), dtype=np.uint8)
        mesh.visual.vertex_colors = colors
        return mesh
    
    async def _generate_procedural_object(self, analysis: Dict[str, Any], style: str, quality: str) -> trimesh.Trimesh:
        """Generate procedural objects for unknown categories"""
        # Use procedural generation based on analysis
        complexity = analysis.get('complexity', 'medium')
        target_vertices = analysis.get('target_vertices', 1000)
        
        # Start with base shape
        if 'round' in analysis.get('enhanced_prompt', '').lower():
            base_mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        elif 'angular' in analysis.get('enhanced_prompt', '').lower():
            base_mesh = trimesh.creation.box(extents=[2, 2, 2])
        else:
            base_mesh = trimesh.creation.cylinder(radius=0.8, height=2.0, sections=12)
        
        # Apply deformations based on complexity
        if complexity in ['high', 'ultra']:
            base_mesh = self._apply_procedural_deformations(base_mesh, analysis)
        
        # Apply colors
        base_mesh = self._apply_generic_colors(base_mesh, analysis)
        
        return base_mesh
    
    def _apply_procedural_deformations(self, mesh: trimesh.Trimesh, analysis: Dict[str, Any]) -> trimesh.Trimesh:
        """Apply procedural deformations to create interesting shapes"""
        enhanced_prompt = analysis.get('enhanced_prompt', '').lower()
        
        if 'twisted' in enhanced_prompt or 'spiral' in enhanced_prompt:
            # Apply twist deformation
            vertices = mesh.vertices.copy()
            for i, vertex in enumerate(vertices):
                height = vertex[2]
                twist_angle = height * np.pi * 0.5
                cos_a, sin_a = np.cos(twist_angle), np.sin(twist_angle)
                x, y = vertex[0], vertex[1]
                vertices[i][0] = x * cos_a - y * sin_a
                vertices[i][1] = x * sin_a + y * cos_a
            mesh.vertices = vertices
        
        if 'spiky' in enhanced_prompt or 'rough' in enhanced_prompt:
            # Add surface noise
            displacement = 0.1 * np.random.random(len(mesh.vertices))
            mesh.vertices += mesh.vertex_normals * displacement.reshape(-1, 1)
        
        if 'smooth' in enhanced_prompt:
            mesh = mesh.smoothed()
        
        return mesh
    
    def _apply_generic_colors(self, mesh: trimesh.Trimesh, analysis: Dict[str, Any]) -> trimesh.Trimesh:
        """Apply generic colors based on analysis"""
        colors = analysis.get('colors', ['gray'])
        primary_color = colors[0] if colors else 'gray'
        
        color_map = {
            'red': [200, 50, 50, 255],
            'blue': [50, 100, 200, 255],
            'green': [50, 150, 50, 255],
            'yellow': [200, 200, 50, 255],
            'purple': [150, 50, 200, 255],
            'orange': [200, 100, 50, 255],
            'white': [240, 240, 240, 255],
            'black': [40, 40, 40, 255],
            'gray': [120, 120, 120, 255]
        }
        
        base_color = color_map.get(primary_color, color_map['gray'])
        colors = np.array([base_color] * len(mesh.vertices), dtype=np.uint8)
        
        # Add variation
        variation = np.random.randint(-20, 20, (len(mesh.vertices), 3))
        colors[:, :3] = np.clip(colors[:, :3] + variation, 0, 255)
        
        mesh.visual.vertex_colors = colors
        return mesh
    
    def _apply_advanced_post_processing(self, mesh: trimesh.Trimesh, analysis: Dict[str, Any], style: str) -> trimesh.Trimesh:
        """Apply advanced post-processing based on style and analysis"""
        try:
            # Basic cleanup
            mesh.remove_degenerate_faces()
            mesh.fix_normals()
            mesh.merge_vertices()
            
            # Style-specific processing
            if style == 'lowpoly':
                # Reduce polygon count while maintaining shape
                mesh = mesh.simplify_quadric_decimation(face_count=len(mesh.faces) // 2)
            elif style == 'stylized':
                # Apply stylized smoothing
                mesh = mesh.smoothed()
            elif style == 'realistic':
                # Add subtle surface details
                if analysis.get('complexity') in ['high', 'ultra']:
                    mesh = self._add_surface_details(mesh, analysis)
            
            # Optimize mesh
            mesh.remove_unreferenced_vertices()
            
            # Ensure proper scaling
            bounds = mesh.bounds
            scale = np.max(bounds[1] - bounds[0])
            if scale > 0:
                target_scale = 2.0  # Target size
                mesh.apply_scale(target_scale / scale)
            
            return mesh
            
        except Exception as e:
            logger.warning(f"⚠️ Post-processing failed: {e}")
            return mesh
    
    def _add_surface_details(self, mesh: trimesh.Trimesh, analysis: Dict[str, Any]) -> trimesh.Trimesh:
        """Add surface details for realistic rendering"""
        # Simple bump mapping effect
        vertices = mesh.vertices.copy()
        for i, vertex in enumerate(vertices):
            # Add small noise to create surface texture
            noise = 0.02 * (np.random.random(3) - 0.5)
            vertices[i] += noise
        mesh.vertices = vertices
        
        return mesh
    
    def _calculate_quality_score(self, mesh: trimesh.Trimesh, analysis: Dict[str, Any]) -> float:
        """Calculate quality score based on mesh properties and analysis"""
        try:
            # Base score
            score = 6.0
            
            # Vertex count relative to target
            target_vertices = analysis.get('target_vertices', 1000)
            actual_vertices = len(mesh.vertices)
            vertex_ratio = min(actual_vertices / max(target_vertices, 1), 2.0)
            score += min(vertex_ratio, 2.0)  # Max 2 points
            
            # Mesh validity
            if mesh.is_watertight:
                score += 1.0
            if mesh.is_winding_consistent:
                score += 1.0
            if mesh.is_empty:
                score -= 3.0
                
            # Complexity match
            complexity = analysis.get('complexity', 'medium')
            if complexity == 'ultra' and actual_vertices > 3000:
                score += 2.0
            elif complexity == 'high' and actual_vertices > 1500:
                score += 1.5
            elif complexity == 'medium' and actual_vertices > 800:
                score += 1.0
            elif complexity == 'low' and actual_vertices < 1000:
                score += 0.5
                
            return min(max(score, 0.0), 10.0)  # Clamp between 0 and 10
            
        except Exception as e:
            logger.warning(f"⚠️ Quality score calculation failed: {e}")
            return 6.0
    
    def _calculate_complexity_score(self, mesh: trimesh.Trimesh, analysis: Dict[str, Any]) -> float:
        """Calculate complexity score"""
        try:
            vertices = len(mesh.vertices)
            faces = len(mesh.faces)
            
            # Base complexity
            complexity = vertices / 1000.0 + faces / 2000.0
            
            # Adjust based on analysis
            if analysis.get('complexity') == 'ultra':
                complexity *= 1.5
            elif analysis.get('complexity') == 'high':
                complexity *= 1.2
            
            return min(complexity, 10.0)
        except:
            return 5.0
    
    def _calculate_accuracy_score(self, mesh: trimesh.Trimesh, analysis: Dict[str, Any], prompt: str) -> float:
        """Estimate accuracy based on prompt matching"""
        try:
            score = 6.0
            
            # Check if object type matches
            object_type = analysis.get('object_type', 'unknown')
            if object_type != 'unknown':
                score += 1.0
            
            # Check if materials match
            materials = analysis.get('materials', [])
            if materials:
                for material in materials:
                    if material in prompt.lower():
                        score += 0.5
            
            # Check if colors match
            colors = analysis.get('colors', [])
            if colors:
                for color in colors:
                    if color in prompt.lower():
                        score += 0.5
            
            return min(score, 10.0)
        except:
            return 6.0
    
    def _update_performance_metrics(self, generation_time: float, quality_score: float, complexity_score: float):
        """Update performance tracking metrics"""
        self.performance_metrics["models_generated"] += 1
        
        # Update average generation time
        total_models = self.performance_metrics["models_generated"]
        current_avg = self.performance_metrics["avg_generation_time"]
        self.performance_metrics["avg_generation_time"] = (
            (current_avg * (total_models - 1) + generation_time) / total_models
        )
        
        # Update extremes
        if generation_time < self.performance_metrics["fastest_generation"]:
            self.performance_metrics["fastest_generation"] = generation_time
        
        if generation_time > self.performance_metrics["slowest_generation"]:
            self.performance_metrics["slowest_generation"] = generation_time
        
        # Track high-quality generations
        if quality_score > 8.0:
            self.performance_metrics["accuracy_improvements"] += 1
        
        if complexity_score > 7.0:
            self.performance_metrics["complex_assets_generated"] += 1
    
    def _create_intelligent_fallback(self, prompt: str, quality: str) -> trimesh.Trimesh:
        """Create a sophisticated fallback when generation fails"""
        # Try to determine what kind of object was requested
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['vehicle', 'car', 'truck', 'bus']):
            return self._generate_generic_vehicle({}, "realistic", quality)
        elif any(word in prompt_lower for word in ['building', 'house', 'structure']):
            return self._generate_generic_building({}, "realistic", quality)
        elif any(word in prompt_lower for word in ['nature', 'tree', 'plant']):
            return self._generate_generic_natural_object({}, "realistic", quality)
        else:
            # Create an abstract artistic representation
            mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
            
            # Apply interesting deformation
            vertices = mesh.vertices.copy()
            for i, vertex in enumerate(vertices):
                distance = np.linalg.norm(vertex)
                displacement = 0.3 * math.sin(5 * distance)
                vertices[i] = vertex * (1 + displacement)
            mesh.vertices = vertices
            
            return mesh
    
    async def mesh_to_file_optimized(self, mesh: trimesh.Trimesh, format_type: str) -> str:
        """Optimized mesh to file conversion"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._mesh_to_file_sync_optimized, mesh, format_type)
    
    def _mesh_to_file_sync_optimized(self, mesh: trimesh.Trimesh, format_type: str) -> str:
        """Synchronous optimized file conversion"""
        suffix = f'.{format_type}'
        
        # Ensure outputs directory exists
        os.makedirs('outputs', exist_ok=True)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir='outputs') as tmp:
            try:
                # Export with optimized settings
                if format_type == 'glb':
                    # GLB export with compression
                    mesh.export(tmp.name, file_type='glb')
                elif format_type == 'obj':
                    # OBJ export
                    mesh.export(tmp.name, file_type='obj', include_texture=False)
                else:
                    # Default export
                    mesh.export(tmp.name, file_type=format_type)
                
                return tmp.name
                
            except Exception as e:
                logger.error(f"❌ Export failed: {e}")
                # Try basic export
                mesh.export(tmp.name, file_type=format_type)
                return tmp.name
    
    async def get_mesh_preview_ultra_fast(self, mesh: trimesh.Trimesh) -> Optional[str]:
        """Ultra-fast preview generation"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._get_mesh_preview_optimized, mesh)
        except Exception as e:
            logger.warning(f"⚠️ Preview generation failed: {e}")
            return None
    
    def _get_mesh_preview_optimized(self, mesh: trimesh.Trimesh) -> str:
        """Optimized preview generation"""
        try:
            # Create scene with optimized settings
            scene = trimesh.Scene(mesh)
            
            # Quick camera setup
            bounds = mesh.bounds
            center = mesh.centroid
            scale = np.max(bounds[1] - bounds[0])
            
            # Position camera for good view
            camera_distance = scale * 2.5
            camera_pos = center + np.array([camera_distance * 0.7, camera_distance * 0.7, camera_distance])
            
            # Set camera transform
            scene.camera_transform = trimesh.transformations.compose_matrix(
                translate=camera_pos
            )
            
            # Point camera at center
            scene.camera.look_at([center], [camera_pos])
            
            # Render with lower resolution for speed
            png_data = scene.save_image(resolution=[400, 400], visible=True)
            
            if png_data:
                return base64.b64encode(png_data).decode('utf-8')
            else:
                return ""
                
        except Exception as e:
            logger.warning(f"⚠️ Preview rendering failed: {e}")
            return ""

# Initialize the optimized generator
generator = AdvancedAI3DAssetGenerator()

# Optimized file storage with automatic cleanup
file_storage: Dict[str, Dict[str, Any]] = {}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Main page with optimized loading"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/generate3d", response_model=GenerationResponse)
async def generate3d_ultra_fast(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Ultra-fast 3D model generation endpoint"""
    generation_start = time.time()
    
    try:
        logger.info(f"🚀 Starting generation for: {request.prompt[:50]}...")
        
        # Generate the model with ultra-fast method
        mesh, analysis, generation_time = await generator.generate_advanced_3d_asset(
            request.prompt, 
            request.format, 
            request.quality, 
            request.style
        )
        
        # Calculate quality scores
        quality_score = generator._calculate_quality_score(mesh, analysis)
        complexity_score = generator._calculate_complexity_score(mesh, analysis)
        accuracy_score = generator._calculate_accuracy_score(mesh, analysis, request.prompt)
        
        # Generate preview in parallel with file conversion
        preview_task = asyncio.create_task(generator.get_mesh_preview_ultra_fast(mesh))
        file_task = asyncio.create_task(generator.mesh_to_file_optimized(mesh, request.format))
        
        # Wait for both tasks
        preview_image, file_path = await asyncio.gather(preview_task, file_task)
        
        # Setup download
        filename = os.path.basename(file_path)
        file_url = f"/download/{request.format}/{filename}"
        
        # Store file info with metadata
        file_storage[filename] = {
            'path': file_path,
            'created': time.time(),
            'format': request.format,
            'prompt': request.prompt
        }
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_file_optimized, filename, 1800)  # 30 minutes
        
        total_time = time.time() - generation_start
        
        logger.info(f"✅ Generation complete in {total_time:.2f}s - Quality: {quality_score:.1f}/10")
        
        return GenerationResponse(
            success=True,
            enhanced_prompt=analysis.get('enhanced_prompt', request.prompt),
            generation_time=round(total_time, 2),
            vertices=len(mesh.vertices),
            faces=len(mesh.faces),
            file_url=file_url,
            preview_image=preview_image,
            format=request.format,
            quality_score=round(quality_score, 1),
            complexity_score=round(complexity_score, 1),
            accuracy_score=round(accuracy_score, 1),
            style_applied=request.style,
            generation_method=analysis.get('category', 'procedural')
        )
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {e}")
        raise HTTPException(status_code=500, detail={
            "error": "Generation failed",
            "detail": str(e),
            "suggestion": "Try a simpler prompt or check system status"
        })

@app.get("/api/status", response_model=StatusResponse)
async def get_status_optimized():
    """Optimized status endpoint"""
    uptime = datetime.now() - generator.app_start_time
    
    # Calculate cache efficiency
    total_requests = generator.performance_metrics.get("models_generated", 0)
    cache_hits = generator.performance_metrics.get("total_cache_hits", 0)
    cache_hit_rate = (cache_hits / max(total_requests, 1)) * 100 if total_requests > 0 else 0
    
    # Get last generation info
    last_generation = None
    if generator.generation_history:
        last_generation = generator.generation_history[-1]
        # Simplify for response
        last_generation = {
            'prompt': last_generation['prompt'][:50] + "..." if len(last_generation['prompt']) > 50 else last_generation['prompt'],
            'generation_time': last_generation['generation_time'],
            'vertices': last_generation['vertices'],
            'faces': last_generation['faces'],
            'object_type': last_generation['analysis'].get('object_type', 'unknown')
        }
    
    return StatusResponse(
        status="online" if generator.model_loaded else "loading",
        model_loaded=generator.model_loaded,
        device=generator.device,
        model_name=generator.model_name,
        total_generations=len(generator.generation_history),
        last_generation=last_generation,
        version="4.0.0",
        uptime=f"{uptime.days}d {uptime.seconds//3600}h {(uptime.seconds//60)%60}m",
        features=[
            "⚡ AI-Powered 3D Asset Generation",
            "🧠 Advanced Prompt Analysis",
            "🏗️ Template-based Generation System",
            "🎨 Material & Texture System",
            "📊 Quality Metrics",
            "🔄 Background Processing",
            "💾 File Management",
            f"📈 {cache_hit_rate:.1f}% Cache Hit Rate"
        ],
        performance_metrics=generator.performance_metrics,
        ai_capabilities={
            "llm_available": generator.llm is not None,
            "object_recognition": True,
            "complex_shape_generation": True
        }
    )

@app.get("/api/history")
async def get_history_optimized():
    """Get optimized generation history"""
    recent_history = generator.generation_history[-20:]  # Last 20 generations
    
    # Simplify history for response
    simplified_history = []
    for gen in recent_history:
        simplified_history.append({
            'timestamp': gen['timestamp'],
            'prompt': gen['prompt'][:70] + "..." if len(gen['prompt']) > 70 else gen['prompt'],
            'object_type': gen['analysis'].get('object_type', 'unknown'),
            'generation_time': gen['generation_time'],
            'vertices': gen['vertices'],
            'faces': gen['faces'],
            'quality_score': gen.get('quality_score', 0)
        })
    
    return {
        "history": simplified_history,
        "total": len(generator.generation_history),
        "performance_summary": {
            "avg_time": generator.performance_metrics.get("avg_generation_time", 0),
            "total_generated": generator.performance_metrics.get("models_generated", 0),
            "cache_hits": generator.performance_metrics.get("total_cache_hits", 0)
        }
    }

@app.get("/download/{format_type}/{filename}")
async def download_file_optimized(format_type: str, filename: str):
    """Optimized file download with metadata"""
    if filename not in file_storage:
        raise HTTPException(status_code=404, detail="File not found or expired")
    
    file_info = file_storage[filename]
    file_path = file_info['path']
    
    if not os.path.exists(file_path):
        # Clean up broken reference
        del file_storage[filename]
        raise HTTPException(status_code=404, detail="File no longer exists")
    
    # Add download tracking
    file_info['downloads'] = file_info.get('downloads', 0) + 1
    file_info['last_download'] = time.time()
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream',
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "X-Generated-From": file_info.get('prompt', 'Unknown')[:100]
        }
    )

@app.get("/api/examples")
async def get_examples_optimized():
    """Get curated example prompts for best results"""
    examples = [
        # Simple shapes
        "A red metallic cube with smooth edges",
        "A blue glass sphere with reflective surface",
        "A wooden cylinder with natural grain texture",
        "A golden pyramid with sharp geometric edges",
        "A purple crystal torus with faceted surface",
        
        # Complex objects
        "A futuristic silver sports car with aerodynamic design",
        "A cozy wooden house with red brick chimney",
        "A tall oak tree with detailed branches and leaves",
        "A delicate pink rose with layered petals",
        "A sleek modern chair with curved metal frame",
        
        # Creative designs
        "A twisted spiral tower with metallic finish",
        "A spiky crystalline formation in emerald green",
        "A smooth organic sculpture with flowing curves",
        "A geometric diamond with brilliant cut facets",
        "A steampunk mechanical gear assembly",
        
        # Abstract forms
        "An abstract wave pattern in ocean blue",
        "A fractal tree structure with recursive branches",
        "A minimalist architectural column design",
        "A textured stone monument with weathered surface",
        "A colorful wind chime with multiple elements"
    ]
    
    return {"examples": examples}

async def cleanup_file_optimized(filename: str, delay: int):
    """Optimized file cleanup with logging"""
    await asyncio.sleep(delay)
    
    if filename in file_storage:
        file_info = file_storage[filename]
        try:
            if os.path.exists(file_info['path']):
                os.unlink(file_info['path'])
            
            del file_storage[filename]
            logger.info(f"🧹 Cleaned up file: {filename}")
            
        except Exception as e:
            logger.error(f"❌ Cleanup failed for {filename}: {e}")

# Startup optimization
@app.on_event("startup")
async def startup_event_optimized():
    """Optimized startup sequence"""
    logger.info("🚀 Starting Advanced AI 3D Asset Generator v4.0...")
    
    # Create directories
    for directory in ['templates', 'static', 'outputs']:
        os.makedirs(directory, exist_ok=True)
    
    # Force garbage collection
    gc.collect()
    
    # Log system info
    logger.info(f"💻 Device: {generator.device}")
    logger.info(f"🤖 LLM: {'✅ Groq Ready' if generator.llm else '❌ Disabled'}")
    logger.info(f"🎯 Asset templates: {len(generator.asset_templates)} loaded")
    logger.info("🌐 Server ready - Professional-grade generation enabled!")
    logger.info("📚 API docs: /docs | ReDoc: /redoc")

@app.on_event("shutdown")
async def shutdown_event_optimized():
    """Optimized shutdown with cleanup"""
    logger.info("🛑 Shutting down Advanced AI 3D Generator...")
    
    # Clean up all files
    cleanup_count = 0
    for filename, file_info in file_storage.items():
        try:
            if os.path.exists(file_info['path']):
                os.unlink(file_info['path'])
                cleanup_count += 1
        except Exception as e:
            logger.error(f"❌ Cleanup error for {filename}: {e}")
    
    logger.info(f"🧹 Cleaned up {cleanup_count} files")
    logger.info("✅ Shutdown complete")

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for better error responses"""
    logger.error(f"❌ Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "The server encountered an unexpected error",
            "suggestion": "Please try again or contact support if the issue persists"
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Disabled for production performance
        log_level="info",
        access_log=True,
        workers=1,  # Single worker for optimal memory usage
        loop="asyncio"  # Use asyncio loop for better performance
    )