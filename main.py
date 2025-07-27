from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
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

# Updated imports with better error handling
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import optimized generation libraries
try:
    from transformers import AutoTokenizer, AutoModel
    import requests
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure optimized logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Pydantic Models
class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000, description="Text description of the 3D model")
    format: str = Field(default="glb", pattern="^(glb|obj)$", description="Output format: glb or obj")

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
    
    model_config = {"protected_namespaces": ()}

# FastAPI App with optimized settings
app = FastAPI(
    title="Ultra-Fast 3D Model Generation ML Tool",
    description="Lightning-Fast AI-Powered 3D Asset Creation - Optimized for Speed & Quality",
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Optimized CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class OptimizedAdvanced3DModelGenerator:
    def __init__(self):
        self.model_loaded = False
        self.model_name = "optimized-parametric-plus"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generation_history = []
        self.app_start_time = datetime.now()
        self.generation_queue = queue.Queue(maxsize=10)
        self.worker_threads = 2  # Number of worker threads
        self.generation_cache = {}  # Cache for repeated prompts
        self.performance_metrics = {
            "avg_generation_time": 0.0,
            "total_cache_hits": 0,
            "models_generated": 0,
            "fastest_generation": float('inf'),
            "slowest_generation": 0.0
        }
        
        # Initialize optimized LangChain with Groq
        self._initialize_llm()
        
        # Start worker threads
        self._start_workers()
        
        # Preload common shapes for instant generation
        self._preload_base_shapes()
        
    def _initialize_llm(self):
        """Initialize LLM with optimized settings"""
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            try:
                self.llm = ChatGroq(
                    api_key=groq_api_key,
                    model="llama3-8b-8192",  # Faster model
                    temperature=0.5,  # Reduced for consistency
                    max_tokens=200,   # Limit for speed
                    timeout=10        # 10 second timeout
                )
                
                # Ultra-optimized prompt template
                self.prompt_template = PromptTemplate(
                    input_variables=["user_prompt"],
                    template="""Convert to 3D specs: "{user_prompt}"

Output format:
Shape: [basic geometric form]
Details: [key visual features]
Scale: [dimensions/proportions]
Style: [aesthetic direction]

Be specific, concise, visual-focused."""
                )
                
                self.chain = self.prompt_template | self.llm | StrOutputParser()
                logger.info("✅ Optimized LangChain with Groq initialized")
                
            except Exception as e:
                logger.warning(f"⚠️ Groq LLM initialization failed: {e}")
                self.llm = None
                self.chain = None
        else:
            logger.warning("⚠️ GROQ_API_KEY not found")
            self.llm = None
            self.chain = None
    
    def _preload_base_shapes(self):
        """Preload common 3D shapes for instant access"""
        logger.info("🔄 Preloading base shapes...")
        
        self.base_shapes = {
            'cube': self._create_optimized_cube(),
            'sphere': self._create_optimized_sphere(),
            'cylinder': self._create_optimized_cylinder(),
            'pyramid': self._create_optimized_pyramid(),
            'torus': self._create_optimized_torus(),
            'cone': self._create_optimized_cone(),
            'capsule': self._create_optimized_capsule(),
            'diamond': self._create_optimized_diamond(),
        }
        
        # Mark as loaded
        self.model_loaded = True
        logger.info("✅ Base shapes preloaded successfully")
    
    def _start_workers(self):
        """Start background worker threads for processing"""
        for i in range(self.worker_threads):
            thread = threading.Thread(target=self._worker, daemon=True)
            thread.start()
            logger.info(f"🔄 Started worker thread {i+1}")
    
    def _worker(self):
        """Background worker for processing generation requests"""
        while True:
            try:
                task = self.generation_queue.get(timeout=1)
                if task is None:
                    break
                
                # Process the task
                task['result'] = self._process_generation_task(task)
                self.generation_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"❌ Worker error: {e}")
    
    def _process_generation_task(self, task):
        """Process a single generation task"""
        try:
            prompt = task['prompt']
            
            # Check cache first
            cache_key = f"{prompt}_{task.get('seed', 0)}"
            if cache_key in self.generation_cache:
                self.performance_metrics["total_cache_hits"] += 1
                return self.generation_cache[cache_key]
            
            # Generate new model
            result = self._generate_optimized_mesh(prompt)
            
            # Cache result
            self.generation_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Task processing error: {e}")
            return None
    
    async def enhance_prompt_ultra_fast(self, user_prompt: str) -> str:
        """Ultra-fast prompt enhancement with timeout"""
        if not self.chain:
            return self._enhance_prompt_locally(user_prompt)
            
        try:
            # Use asyncio.wait_for with timeout
            enhanced_prompt = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.chain.invoke({"user_prompt": user_prompt})
                ),
                timeout=5.0  # 5 second timeout
            )
            return enhanced_prompt.strip()
            
        except asyncio.TimeoutError:
            logger.warning("⚠️ LLM timeout, using local enhancement")
            return self._enhance_prompt_locally(user_prompt)
        except Exception as e:
            logger.error(f"❌ LLM error: {e}")
            return self._enhance_prompt_locally(user_prompt)
    
    def _enhance_prompt_locally(self, prompt: str) -> str:
        """Fast local prompt enhancement"""
        prompt_lower = prompt.lower()
        
        # Extract key features
        shapes = []
        materials = []
        colors = []
        
        # Shape detection
        shape_keywords = {
            'cube': ['cube', 'box', 'block', 'square'],
            'sphere': ['sphere', 'ball', 'globe', 'round'],
            'cylinder': ['cylinder', 'tube', 'pipe', 'column'],
            'pyramid': ['pyramid', 'triangle', 'peak', 'cone'],
            'car': ['car', 'vehicle', 'automobile'],
            'house': ['house', 'building', 'home'],
            'tree': ['tree', 'plant', 'organic'],
            'flower': ['flower', 'rose', 'bloom']
        }
        
        for shape, keywords in shape_keywords.items():
            if any(kw in prompt_lower for kw in keywords):
                shapes.append(shape)
        
        # Color detection
        color_keywords = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white']
        for color in color_keywords:
            if color in prompt_lower:
                colors.append(color)
        
        # Material detection
        material_keywords = ['metal', 'wood', 'glass', 'plastic', 'stone', 'ceramic']
        for material in material_keywords:
            if material in prompt_lower:
                materials.append(material)
        
        # Build enhanced prompt
        enhanced = f"3D Model: {prompt}. "
        if shapes:
            enhanced += f"Primary shape: {shapes[0]}. "
        if colors:
            enhanced += f"Color: {colors[0]}. "
        if materials:
            enhanced += f"Material: {materials[0]}. "
        
        return enhanced
    
    def _generate_optimized_mesh(self, enhanced_prompt: str) -> trimesh.Trimesh:
        """Generate optimized 3D mesh based on enhanced prompt"""
        prompt_lower = enhanced_prompt.lower()
        
        # Smart shape selection with better algorithms
        if any(word in prompt_lower for word in ['cube', 'box', 'block', 'building']):
            base_mesh = self.base_shapes['cube'].copy()
            mesh = self._enhance_cube(base_mesh, prompt_lower)
        elif any(word in prompt_lower for word in ['sphere', 'ball', 'planet', 'globe']):
            base_mesh = self.base_shapes['sphere'].copy()
            mesh = self._enhance_sphere(base_mesh, prompt_lower)
        elif any(word in prompt_lower for word in ['cylinder', 'tube', 'pillar', 'column']):
            base_mesh = self.base_shapes['cylinder'].copy()
            mesh = self._enhance_cylinder(base_mesh, prompt_lower)
        elif any(word in prompt_lower for word in ['pyramid', 'triangle', 'peak']):
            base_mesh = self.base_shapes['pyramid'].copy()
            mesh = self._enhance_pyramid(base_mesh, prompt_lower)
        elif any(word in prompt_lower for word in ['torus', 'ring', 'donut']):
            base_mesh = self.base_shapes['torus'].copy()
            mesh = self._enhance_torus(base_mesh, prompt_lower)
        elif any(word in prompt_lower for word in ['cone', 'ice cream']):
            base_mesh = self.base_shapes['cone'].copy()
            mesh = self._enhance_cone(base_mesh, prompt_lower)
        elif any(word in prompt_lower for word in ['diamond', 'gem', 'crystal']):
            base_mesh = self.base_shapes['diamond'].copy()
            mesh = self._enhance_diamond(base_mesh, prompt_lower)
        elif any(word in prompt_lower for word in ['car', 'vehicle', 'automobile']):
            mesh = self._create_detailed_car(prompt_lower)
        elif any(word in prompt_lower for word in ['house', 'home', 'building']):
            mesh = self._create_detailed_house(prompt_lower)
        elif any(word in prompt_lower for word in ['tree', 'plant']):
            mesh = self._create_detailed_tree(prompt_lower)
        elif any(word in prompt_lower for word in ['flower', 'rose', 'bloom']):
            mesh = self._create_detailed_flower(prompt_lower)
        else:
            # Create custom shape based on prompt analysis
            mesh = self._create_custom_shape(prompt_lower)
        
        # Apply colors and materials
        mesh = self._apply_materials_and_colors(mesh, prompt_lower)
        
        # Optimize mesh
        mesh = self._optimize_mesh(mesh)
        
        return mesh
    
    # Optimized base shape creators
    def _create_optimized_cube(self) -> trimesh.Trimesh:
        return trimesh.creation.box(extents=[2, 2, 2])
    
    def _create_optimized_sphere(self) -> trimesh.Trimesh:
        return trimesh.creation.uv_sphere(radius=1.0, count=[20, 20])
    
    def _create_optimized_cylinder(self) -> trimesh.Trimesh:
        return trimesh.creation.cylinder(radius=0.8, height=2.0, sections=20)
    
    def _create_optimized_pyramid(self) -> trimesh.Trimesh:
        vertices = np.array([
            [0, 0, 1.5],      # apex
            [-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]  # base
        ])
        faces = np.array([
            [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1],  # sides
            [1, 4, 3], [1, 3, 2]  # base
        ])
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    
    def _create_optimized_torus(self) -> trimesh.Trimesh:
        return trimesh.creation.torus(major_radius=1.0, minor_radius=0.3, major_sections=20, minor_sections=12)
    
    def _create_optimized_cone(self) -> trimesh.Trimesh:
        return trimesh.creation.cone(radius=1.0, height=2.0, sections=20)
    
    def _create_optimized_capsule(self) -> trimesh.Trimesh:
        return trimesh.creation.capsule(radius=0.5, height=2.0)
    
    def _create_optimized_diamond(self) -> trimesh.Trimesh:
        vertices = np.array([
            [0, 0, 1.2],      # top apex
            [0, 0, -1.2],     # bottom apex
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],  # middle ring
            [0.7, 0.7, 0.3], [-0.7, 0.7, 0.3], [-0.7, -0.7, 0.3], [0.7, -0.7, 0.3]  # upper ring
        ])
        faces = np.array([
            # Top faces
            [0, 6, 2], [0, 2, 9], [0, 9, 3], [0, 3, 8], [0, 8, 4], [0, 4, 7], [0, 7, 5], [0, 5, 6],
            # Bottom faces  
            [1, 2, 6], [1, 9, 2], [1, 3, 9], [1, 8, 3], [1, 4, 8], [1, 7, 4], [1, 5, 7], [1, 6, 5]
        ])
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Enhanced shape modifiers
    def _enhance_cube(self, mesh: trimesh.Trimesh, prompt: str) -> trimesh.Trimesh:
        if 'smooth' in prompt:
            mesh = mesh.smoothed()
        if 'large' in prompt or 'big' in prompt:
            mesh.apply_scale(1.5)
        if 'small' in prompt or 'tiny' in prompt:
            mesh.apply_scale(0.6)
        return mesh
    
    def _enhance_sphere(self, mesh: trimesh.Trimesh, prompt: str) -> trimesh.Trimesh:
        if 'spiky' in prompt or 'rough' in prompt:
            # Add noise to vertices
            noise = np.random.normal(0, 0.1, mesh.vertices.shape)
            mesh.vertices += noise
        if 'large' in prompt:
            mesh.apply_scale(1.8)
        return mesh
    
    def _enhance_cylinder(self, mesh: trimesh.Trimesh, prompt: str) -> trimesh.Trimesh:
        if 'tall' in prompt or 'long' in prompt:
            mesh.apply_scale([1, 1, 2])
        if 'wide' in prompt or 'thick' in prompt:
            mesh.apply_scale([1.5, 1.5, 1])
        return mesh
    
    def _enhance_pyramid(self, mesh: trimesh.Trimesh, prompt: str) -> trimesh.Trimesh:
        if 'sharp' in prompt:
            # Make apex sharper
            vertices = mesh.vertices.copy()
            vertices[0][2] *= 1.3  # Make peak higher
            mesh.vertices = vertices
        return mesh
    
    def _enhance_torus(self, mesh: trimesh.Trimesh, prompt: str) -> trimesh.Trimesh:
        if 'thick' in prompt:
            mesh.apply_scale([1, 1, 1.5])
        return mesh
    
    def _enhance_cone(self, mesh: trimesh.Trimesh, prompt: str) -> trimesh.Trimesh:
        if 'sharp' in prompt:
            mesh.apply_scale([1, 1, 1.3])
        return mesh
    
    def _enhance_diamond(self, mesh: trimesh.Trimesh, prompt: str) -> trimesh.Trimesh:
        if 'brilliant' in prompt:
            # Add more facets by subdividing
            mesh = mesh.subdivide()
        return mesh
    
    def _create_detailed_car(self, prompt: str) -> trimesh.Trimesh:
        """Create a detailed car model"""
        # Main body
        body = trimesh.creation.box(extents=[4, 1.8, 1])
        body.apply_translation([0, 0, 0.5])
        
        # Cabin
        cabin = trimesh.creation.box(extents=[2, 1.6, 0.8])
        cabin.apply_translation([0.5, 0, 1.4])
        
        # Wheels
        wheels = []
        wheel_positions = [[-1.3, -1, 0], [-1.3, 1, 0], [1.3, -1, 0], [1.3, 1, 0]]
        for pos in wheel_positions:
            wheel = trimesh.creation.cylinder(radius=0.4, height=0.3)
            wheel.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
            wheel.apply_translation(pos)
            wheels.append(wheel)
        
        # Hood and trunk
        hood = trimesh.creation.box(extents=[1, 1.8, 0.3])
        hood.apply_translation([-1.5, 0, 0.65])
        
        trunk = trimesh.creation.box(extents=[0.8, 1.8, 0.3])
        trunk.apply_translation([1.6, 0, 0.65])
        
        # Combine all parts
        car_parts = [body, cabin, hood, trunk] + wheels
        
        # Color based on prompt
        if 'red' in prompt:
            color = [200, 50, 50, 255]
        elif 'blue' in prompt:
            color = [50, 100, 200, 255]
        else:
            color = [100, 100, 100, 255]
        
        return trimesh.util.concatenate(car_parts)
    
    def _create_detailed_house(self, prompt: str) -> trimesh.Trimesh:
        """Create a detailed house model"""
        # Main structure
        base = trimesh.creation.box(extents=[3, 2.5, 2])
        base.apply_translation([0, 0, 1])
        
        # Roof - create a proper triangular prism
        roof_vertices = np.array([
            [-1.5, -1.25, 2], [1.5, -1.25, 2], [1.5, 1.25, 2], [-1.5, 1.25, 2],  # base
            [0, -1.25, 3.2], [0, 1.25, 3.2]  # peak
        ])
        roof_faces = np.array([
            [0, 1, 4], [1, 2, 5], [2, 3, 5], [3, 0, 4],  # sides
            [4, 5, 1], [4, 1, 0], [5, 4, 3], [5, 3, 2]   # triangular ends
        ])
        roof = trimesh.Trimesh(vertices=roof_vertices, faces=roof_faces)
        
        # Door
        door = trimesh.creation.box(extents=[0.3, 0.8, 1.8])
        door.apply_translation([0, -1.3, 0.9])
        
        # Windows
        window1 = trimesh.creation.box(extents=[0.1, 0.8, 0.8])
        window1.apply_translation([-0.8, -1.3, 1.2])
        
        window2 = trimesh.creation.box(extents=[0.1, 0.8, 0.8])
        window2.apply_translation([0.8, -1.3, 1.2])
        
        # Chimney
        chimney = trimesh.creation.box(extents=[0.4, 0.4, 1])
        chimney.apply_translation([1, 0.8, 3.5])
        
        return trimesh.util.concatenate([base, roof, door, window1, window2, chimney])
    
    def _create_detailed_tree(self, prompt: str) -> trimesh.Trimesh:
        """Create a detailed tree model"""
        # Trunk with taper
        trunk_vertices = []
        trunk_faces = []
        
        # Create tapered trunk
        sections = 8
        for i in range(sections + 1):
            height = i * 2.0 / sections
            radius = 0.15 * (1 - i * 0.1 / sections)  # Taper
            
            for j in range(12):
                angle = j * 2 * np.pi / 12
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                trunk_vertices.append([x, y, height])
        
        # Create faces for trunk
        for i in range(sections):
            for j in range(12):
                next_j = (j + 1) % 12
                
                # Current ring
                v1 = i * 12 + j
                v2 = i * 12 + next_j
                # Next ring
                v3 = (i + 1) * 12 + j
                v4 = (i + 1) * 12 + next_j
                
                # Two triangles per quad
                trunk_faces.extend([[v1, v2, v3], [v2, v4, v3]])
        
        trunk = trimesh.Trimesh(vertices=np.array(trunk_vertices), faces=np.array(trunk_faces))
        
        # Crown - multiple spheres for fuller look
        crown_parts = []
        crown_positions = [
            [0, 0, 2.5], [-0.3, 0.3, 2.3], [0.3, -0.3, 2.3], 
            [0.2, 0.4, 2.8], [-0.2, -0.4, 2.8]
        ]
        
        for pos in crown_positions:
            crown = trimesh.creation.uv_sphere(radius=0.6, count=[12, 12])
            crown.apply_translation(pos)
            crown_parts.append(crown)
        
        # Branches
        branch_positions = [
            [0.8, 0, 1.5], [-0.8, 0, 1.5], [0, 0.8, 1.8], [0, -0.8, 1.8]
        ]
        
        for pos in branch_positions:
            branch = trimesh.creation.cylinder(radius=0.05, height=0.8)
            branch.apply_transform(trimesh.transformations.rotation_matrix(np.pi/3, [1, 0, 0]))
            branch.apply_translation(pos)
            crown_parts.append(branch)
        
        return trimesh.util.concatenate([trunk] + crown_parts)
    
    def _create_detailed_flower(self, prompt: str) -> trimesh.Trimesh:
        """Create a detailed flower model"""
        # Stem
        stem = trimesh.creation.cylinder(radius=0.05, height=2)
        stem.apply_translation([0, 0, 1])
        
        # Center
        center = trimesh.creation.uv_sphere(radius=0.15, count=[12, 12])
        center.apply_translation([0, 0, 2])
        
        # Petals - create detailed petal shapes
        petals = []
        num_petals = 8
        
        for i in range(num_petals):
            angle = i * 2 * np.pi / num_petals
            
            # Create petal as stretched sphere
            petal = trimesh.creation.uv_sphere(radius=0.3, count=[8, 8])
            petal.apply_scale([0.3, 1, 0.1])  # Flatten and elongate
            
            # Position around center
            x = 0.4 * np.cos(angle)
            y = 0.4 * np.sin(angle)
            petal.apply_translation([x, y, 2])
            
            # Rotate to face outward
            petal.apply_transform(trimesh.transformations.rotation_matrix(angle, [0, 0, 1]))
            
            petals.append(petal)
        
        # Leaves
        leaf1 = trimesh.creation.box(extents=[0.6, 0.15, 0.05])
        leaf1.apply_translation([0.3, 0, 0.8])
        
        leaf2 = trimesh.creation.box(extents=[0.15, 0.6, 0.05])
        leaf2.apply_translation([0, 0.3, 1.2])
        
        return trimesh.util.concatenate([stem, center] + petals + [leaf1, leaf2])
    
    def _create_custom_shape(self, prompt: str) -> trimesh.Trimesh:
        """Create custom shape based on prompt analysis"""
        # Start with icosphere for organic shapes
        mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
        
        # Apply deformations based on keywords
        if 'twisted' in prompt or 'spiral' in prompt:
            vertices = mesh.vertices.copy()
            for i, vertex in enumerate(vertices):
                height = vertex[2]
                twist_angle = height * np.pi
                cos_a, sin_a = np.cos(twist_angle), np.sin(twist_angle)
                x, y = vertex[0], vertex[1]
                vertices[i][0] = x * cos_a - y * sin_a
                vertices[i][1] = x * sin_a + y * cos_a
            mesh.vertices = vertices
        
        if 'spiky' in prompt:
            # Add spikes by displacing vertices along normals
            displacement = 0.3 * np.random.random(len(mesh.vertices))
            mesh.vertices += mesh.vertex_normals * displacement.reshape(-1, 1)
        
        if 'smooth' in prompt:
            mesh = mesh.smoothed()
        
        return mesh
    
    def _apply_materials_and_colors(self, mesh: trimesh.Trimesh, prompt: str) -> trimesh.Trimesh:
        """Apply colors and materials based on prompt"""
        # Base color
        if 'red' in prompt:
            color = [200, 50, 50, 255]
        elif 'blue' in prompt:
            color = [50, 100, 200, 255]
        elif 'green' in prompt:
            color = [50, 150, 50, 255]
        elif 'yellow' in prompt:
            color = [200, 200, 50, 255]
        elif 'purple' in prompt:
            color = [150, 50, 200, 255]
        elif 'orange' in prompt:
            color = [200, 100, 50, 255]
        elif 'white' in prompt:
            color = [240, 240, 240, 255]
        elif 'black' in prompt:
            color = [40, 40, 40, 255]
        else:
            color = [120, 120, 120, 255]
        
        # Apply color with slight variation
        colors = np.array([color] * len(mesh.vertices), dtype=np.uint8)
        
        # Add variation for more realistic look
        if 'metallic' in prompt or 'shiny' in prompt:
            # Add metallic variation
            variation = np.random.randint(-20, 20, (len(mesh.vertices), 3))
            colors[:, :3] = np.clip(colors[:, :3] + variation, 50, 255)
        elif 'matte' in prompt or 'dull' in prompt:
            # Reduce brightness for matte look
            colors[:, :3] = (colors[:, :3] * 0.7).astype(np.uint8)
        else:
            # Standard variation
            variation = np.random.randint(-15, 15, (len(mesh.vertices), 3))
            colors[:, :3] = np.clip(colors[:, :3] + variation, 0, 255)
        
        mesh.visual.vertex_colors = colors
        return mesh
    
    def _optimize_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Optimize mesh for performance and quality"""
        try:
            # Remove degenerate faces
            mesh.remove_degenerate_faces()
            
            # Fix normals
            mesh.fix_normals()
            
            # Remove duplicate vertices (but keep it fast)
            mesh.merge_vertices()
            
            # Ensure mesh is watertight if possible
            if not mesh.is_watertight:
                try:
                    mesh.fill_holes()
                except:
                    pass  # If hole filling fails, continue anyway
            
            # Remove unreferenced vertices
            mesh.remove_unreferenced_vertices()
            
            return mesh
            
        except Exception as e:
            logger.warning(f"⚠️ Mesh optimization failed: {e}")
            return mesh
    
    async def generate_3d_model_ultra_fast(self, prompt: str) -> tuple[trimesh.Trimesh, str, float]:
        """Ultra-fast 3D model generation with caching and optimization"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = f"{prompt}_{hash(prompt) % 1000}"
            if cache_key in self.generation_cache:
                cached_result = self.generation_cache[cache_key]
                generation_time = time.time() - start_time
                self.performance_metrics["total_cache_hits"] += 1
                logger.info(f"✅ Cache hit for prompt: {prompt[:50]}...")
                return cached_result[0].copy(), cached_result[1], generation_time
            
            # Enhance prompt with timeout
            enhanced_prompt = await self.enhance_prompt_ultra_fast(prompt)
            
            # Generate mesh using optimized methods
            mesh = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_optimized_mesh, enhanced_prompt
            )
            
            generation_time = time.time() - start_time
            
            # Update performance metrics
            self._update_performance_metrics(generation_time)
            
            # Cache the result
            self.generation_cache[cache_key] = (mesh.copy(), enhanced_prompt)
            
            # Limit cache size
            if len(self.generation_cache) > 50:
                # Remove oldest entries
                oldest_key = next(iter(self.generation_cache))
                del self.generation_cache[oldest_key]
            
            # Log generation
            self.generation_history.append({
                'prompt': prompt,
                'enhanced_prompt': enhanced_prompt,
                'timestamp': datetime.now().isoformat(),
                'generation_time': generation_time,
                'vertices': len(mesh.vertices),
                'faces': len(mesh.faces),
                'cached': False
            })
            
            # Keep history manageable
            if len(self.generation_history) > 100:
                self.generation_history = self.generation_history[-50:]
            
            logger.info(f"✅ Generated model in {generation_time:.2f}s - {len(mesh.vertices)} vertices")
            
            return mesh, enhanced_prompt, generation_time
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            # Return fallback cube
            fallback_mesh = self.base_shapes['cube'].copy()
            generation_time = time.time() - start_time
            return fallback_mesh, f"Fallback for: {prompt}", generation_time
    
    def _update_performance_metrics(self, generation_time: float):
        """Update performance tracking metrics"""
        self.performance_metrics["models_generated"] += 1
        
        # Update average
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
            png_data = scene.save_image(resolution=[300, 300], visible=True)
            
            if png_data:
                return base64.b64encode(png_data).decode('utf-8')
            else:
                return ""
                
        except Exception as e:
            logger.warning(f"⚠️ Preview rendering failed: {e}")
            return ""
    
    def calculate_quality_score(self, mesh: trimesh.Trimesh, prompt: str) -> float:
        """Calculate a quality score for the generated model"""
        try:
            score = 5.0  # Base score
            
            # Vertex count score (optimal range)
            vertex_count = len(mesh.vertices)
            if 100 <= vertex_count <= 10000:
                score += 2.0
            elif vertex_count > 10000:
                score += 1.0  # Too many vertices
            
            # Mesh validity
            if mesh.is_watertight:
                score += 1.5
            if mesh.is_winding_consistent:
                score += 1.0
            
            # Complexity based on prompt
            prompt_lower = prompt.lower()
            complexity_keywords = ['detailed', 'complex', 'intricate', 'elaborate']
            if any(kw in prompt_lower for kw in complexity_keywords):
                if vertex_count > 500:
                    score += 1.0
            
            # Penalize for common issues
            if len(mesh.faces) == 0:
                score -= 3.0
            if vertex_count < 8:
                score -= 2.0  # Too simple
            
            return min(max(score, 0.0), 10.0)  # Clamp between 0 and 10
            
        except Exception as e:
            logger.warning(f"⚠️ Quality score calculation failed: {e}")
            return 5.0

# Initialize the optimized generator
generator = OptimizedAdvanced3DModelGenerator()

# Optimized file storage with automatic cleanup
file_storage: Dict[str, Dict[str, Any]] = {}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Main page with optimized loading"""
    return templates.TemplateResponse("index.html", {"request": {}})

@app.post("/api/generate3d", response_model=GenerationResponse)
async def generate3d_ultra_fast(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Ultra-fast 3D model generation endpoint"""
    generation_start = time.time()
    
    try:
        logger.info(f"🚀 Starting generation for: {request.prompt[:50]}...")
        
        # Generate the model with ultra-fast method
        mesh, enhanced_prompt, generation_time = await generator.generate_3d_model_ultra_fast(request.prompt)
        
        # Calculate quality score
        quality_score = generator.calculate_quality_score(mesh, request.prompt)
        
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
            enhanced_prompt=enhanced_prompt,
            generation_time=round(total_time, 2),
            vertices=len(mesh.vertices),
            faces=len(mesh.faces),
            file_url=file_url,
            preview_image=preview_image,
            format=request.format,
            quality_score=round(quality_score, 1)
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
    cache_hit_rate = (cache_hits / max(total_requests, 1)) * 100
    
    return StatusResponse(
        status="online" if generator.model_loaded else "loading",
        model_loaded=generator.model_loaded,
        device=generator.device,
        model_name=generator.model_name,
        total_generations=len(generator.generation_history),
        last_generation=generator.generation_history[-1] if generator.generation_history else None,
        version="3.0.0",
        uptime=f"{uptime.days}d {uptime.seconds//3600}h {(uptime.seconds//60)%60}m",
        features=[
            "⚡ Ultra-fast generation (<10s)",
            "🧠 AI prompt enhancement" if generator.llm else "📝 Local prompt processing",
            "🚀 Groq LLM integration" if generator.llm else "❌ No LLM integration",
            "📦 Multiple formats (GLB/OBJ)",
            "🖼️ Instant previews",
            "🎯 Smart caching system",
            "⚙️ Optimized parametric generation",
            "🔄 Async processing",
            "📊 Quality scoring",
            "🎨 Advanced materials & colors",
            "🏗️ Complex object support",
            "🧹 Auto-cleanup",
            f"📈 {cache_hit_rate:.1f}% cache hit rate"
        ],
        performance_metrics={
            "avg_generation_time": round(generator.performance_metrics.get("avg_generation_time", 0), 2),
            "fastest_generation": round(generator.performance_metrics.get("fastest_generation", 0), 2) if generator.performance_metrics.get("fastest_generation", float('inf')) != float('inf') else 0,
            "slowest_generation": round(generator.performance_metrics.get("slowest_generation", 0), 2),
            "cache_hit_rate": round(cache_hit_rate, 1),
            "cache_size": len(generator.generation_cache),
            "active_files": len(file_storage)
        }
    )

@app.get("/api/history")
async def get_history_optimized():
    """Get optimized generation history"""
    recent_history = generator.generation_history[-20:]  # Last 20 generations
    
    return {
        "history": recent_history,
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
    logger.info("🚀 Starting Ultra-Fast 3D Model Generator v3.0...")
    
    # Create directories
    for directory in ['templates', 'static', 'outputs']:
        os.makedirs(directory, exist_ok=True)
    
    # Force garbage collection
    gc.collect()
    
    # Log system info
    logger.info(f"💻 Device: {generator.device}")
    logger.info(f"🤖 LLM: {'✅ Groq Ready' if generator.llm else '❌ Disabled'}")
    logger.info(f"🎯 Base shapes: {len(generator.base_shapes)} preloaded")
    logger.info(f"⚡ Worker threads: {generator.worker_threads}")
    logger.info("🌐 Server ready - Ultra-fast generation enabled!")
    logger.info("📚 API docs: /docs | ReDoc: /redoc")

@app.on_event("shutdown")
async def shutdown_event_optimized():
    """Optimized shutdown with cleanup"""
    logger.info("🛑 Shutting down Ultra-Fast 3D Generator...")
    
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
    return HTTPException(
        status_code=500,
        detail={
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