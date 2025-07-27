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

# Updated LangChain imports - using newer patterns
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser

# Hugging Face imports
from transformers import pipeline
try:
    from diffusers import ShapEPipeline
except ImportError:
    print("ShapE not available, will use fallback generation")
    ShapEPipeline = None

import requests

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
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
    
    model_config = {"protected_namespaces": ()}  # Fix for Pydantic warnings

class HistoryResponse(BaseModel):
    history: List[Dict[str, Any]]
    total: int

class ExamplesResponse(BaseModel):
    examples: List[str]

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

# FastAPI App
app = FastAPI(
    title="Advanced 3D Model Generation ML Tool",
    description="AI-Powered 3D Asset Creation with LangChain, Groq & Hugging Face",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

class Advanced3DModelGenerator:
    def __init__(self):
        self.model_loaded = False
        self.model_name = "fallback-parametric"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generation_history = []
        self.app_start_time = datetime.now()
        self.fallback_mode = True
        self._model_loading_task = None  # Store task reference
        
        # Initialize LangChain with Groq - Updated approach
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            try:
                self.llm = ChatGroq(
                    api_key=groq_api_key,
                    model="mixtral-8x7b-32768",
                    temperature=0.7
                )
                
                # Enhanced prompt template for 3D generation
                self.prompt_template = PromptTemplate(
                    input_variables=["user_prompt"],
                    template="""You are an expert 3D modeling assistant. Given the user's request: "{user_prompt}"

Create a detailed, technical description optimized for 3D model generation that includes:
1. Geometric shape and structure
2. Material properties and textures  
3. Scale and proportions
4. Key visual features and details
5. Style and aesthetic direction

Focus on concrete, visual elements that a 3D generation model can understand.
Avoid abstract concepts and emphasize physical characteristics.

Enhanced 3D Description:"""
                )
                
                # Create chain using newer pattern
                self.chain = self.prompt_template | self.llm | StrOutputParser()
                logger.info("LangChain with Groq initialized successfully")
                
            except Exception as e:
                logger.warning(f"Could not initialize Groq LLM: {e}")
                self.llm = None
                self.chain = None
        else:
            logger.warning("GROQ_API_KEY not found, prompt enhancement disabled")
            self.llm = None
            self.chain = None
    
    def start_model_loading(self):
        """Start model loading when event loop is available"""
        try:
            loop = asyncio.get_running_loop()
            self._model_loading_task = loop.create_task(self.load_model_async())
        except RuntimeError:
            # Event loop not running yet, will be called from startup event
            pass
    
    async def load_model_async(self):
        """Load the Shap-E model asynchronously"""
        try:
            logger.info(f"Attempting to load Shap-E model on {self.device}...")
            
            if ShapEPipeline is None:
                logger.info("ShapE not available, using parametric fallback")
                self.model_loaded = True
                return
            
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._load_model_sync)
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Falling back to parametric generation")
            self.model_loaded = True
            self.fallback_mode = True
    
    def _load_model_sync(self):
        """Synchronous model loading"""
        try:
            self.pipe = ShapEPipeline.from_pretrained(
                "openai/shap-e",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                variant="fp16" if self.device == "cuda" else None
            )
            self.pipe = self.pipe.to(self.device)
            logger.info("Shap-E model loaded successfully!")
            self.model_loaded = True
            self.model_name = "openai/shap-e"
            self.fallback_mode = False
        except Exception as e:
            logger.warning(f"Shap-E not available, using fallback: {e}")
            self.model_loaded = True
            self.fallback_mode = True
    
    async def enhance_prompt_with_llm(self, user_prompt: str) -> str:
        """Use LangChain + Groq to enhance the user prompt asynchronously"""
        if not self.chain:
            logger.info("LLM not available, using original prompt")
            return user_prompt
            
        try:
            loop = asyncio.get_event_loop()
            enhanced_prompt = await loop.run_in_executor(
                None, lambda: self.chain.invoke({"user_prompt": user_prompt})
            )
            logger.info(f"Enhanced prompt: {enhanced_prompt[:100]}...")
            return enhanced_prompt.strip()
        except Exception as e:
            logger.error(f"Error enhancing prompt: {e}")
            return user_prompt
    
    def generate_parametric_3d(self, prompt: str) -> trimesh.Trimesh:
        """Generate 3D models using parametric shapes based on prompt analysis"""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ['cube', 'box', 'block', 'building']):
            mesh = self.create_enhanced_cube()
        elif any(word in prompt_lower for word in ['sphere', 'ball', 'planet', 'globe']):
            mesh = self.create_enhanced_sphere()
        elif any(word in prompt_lower for word in ['cylinder', 'tube', 'pillar', 'column']):
            mesh = self.create_enhanced_cylinder()
        elif any(word in prompt_lower for word in ['pyramid', 'triangle', 'peak']):
            mesh = self.create_pyramid()
        elif any(word in prompt_lower for word in ['torus', 'ring', 'donut']):
            mesh = self.create_torus()
        elif any(word in prompt_lower for word in ['tree', 'plant', 'organic']):
            mesh = self.create_stylized_tree()
        elif any(word in prompt_lower for word in ['house', 'home', 'building']):
            mesh = self.create_simple_house()
        elif any(word in prompt_lower for word in ['car', 'vehicle', 'automobile']):
            mesh = self.create_simple_car()
        elif any(word in prompt_lower for word in ['flower', 'rose', 'bloom']):
            mesh = self.create_flower()
        else:
            mesh = self.create_fractal_shape()
        
        return mesh
    
    def create_enhanced_cube(self) -> trimesh.Trimesh:
        """Create a stylized cube with details"""
        mesh = trimesh.creation.box(extents=[2, 2, 2])
        return mesh.smoothed()
    
    def create_enhanced_sphere(self) -> trimesh.Trimesh:
        """Create a detailed sphere"""
        return trimesh.creation.uv_sphere(radius=1.0, count=[32, 32])
    
    def create_enhanced_cylinder(self) -> trimesh.Trimesh:
        """Create a detailed cylinder"""
        return trimesh.creation.cylinder(radius=0.8, height=2.0, sections=32)
    
    def create_pyramid(self) -> trimesh.Trimesh:
        """Create a pyramid"""
        vertices = np.array([
            [0, 0, 1.5],      # apex
            [-1, -1, 0],      # base corners
            [1, -1, 0],
            [1, 1, 0],
            [-1, 1, 0]
        ])
        
        faces = np.array([
            [0, 1, 2],  [0, 2, 3],  [0, 3, 4],  [0, 4, 1],
            [1, 4, 3],  [1, 3, 2]
        ])
        
        return trimesh.Trimesh(vertices=vertices, faces=faces)
    
    def create_torus(self) -> trimesh.Trimesh:
        """Create a torus"""
        return trimesh.creation.torus(major_radius=1.0, minor_radius=0.3, major_sections=32, minor_sections=16)
    
    def create_stylized_tree(self) -> trimesh.Trimesh:
        """Create a simple stylized tree"""
        trunk = trimesh.creation.cylinder(radius=0.1, height=1.0)
        trunk.apply_translation([0, 0, 0.5])
        
        crown = trimesh.creation.uv_sphere(radius=0.8)
        crown.apply_translation([0, 0, 1.5])
        
        return trimesh.util.concatenate([trunk, crown])
    
    def create_simple_house(self) -> trimesh.Trimesh:
        """Create a simple house structure"""
        base = trimesh.creation.box(extents=[2, 1.5, 1])
        base.apply_translation([0, 0, 0.5])
        
        # Create roof as a prism
        roof_vertices = np.array([
            [-1, -0.75, 1], [1, -0.75, 1], [1, 0.75, 1], [-1, 0.75, 1], 
            [0, -0.75, 1.8], [0, 0.75, 1.8]
        ])
        roof_faces = np.array([
            [0, 1, 4], [1, 2, 5], [2, 3, 5], [3, 0, 4], 
            [0, 4, 5], [0, 5, 3], [1, 5, 4], [1, 2, 5]
        ])
        roof = trimesh.Trimesh(vertices=roof_vertices, faces=roof_faces)
        
        return trimesh.util.concatenate([base, roof])
    
    def create_simple_car(self) -> trimesh.Trimesh:
        """Create a simple car shape"""
        # Car body
        body = trimesh.creation.box(extents=[3, 1.5, 0.8])
        body.apply_translation([0, 0, 0.4])
        
        # Car roof
        roof = trimesh.creation.box(extents=[1.5, 1.2, 0.6])
        roof.apply_translation([0, 0, 1.1])
        
        # Wheels
        wheel1 = trimesh.creation.cylinder(radius=0.3, height=0.2)
        wheel1.apply_translation([-1, -0.9, 0])
        
        wheel2 = trimesh.creation.cylinder(radius=0.3, height=0.2)
        wheel2.apply_translation([1, -0.9, 0])
        
        wheel3 = trimesh.creation.cylinder(radius=0.3, height=0.2)
        wheel3.apply_translation([-1, 0.9, 0])
        
        wheel4 = trimesh.creation.cylinder(radius=0.3, height=0.2)
        wheel4.apply_translation([1, 0.9, 0])
        
        return trimesh.util.concatenate([body, roof, wheel1, wheel2, wheel3, wheel4])
    
    def create_flower(self) -> trimesh.Trimesh:
        """Create a simple flower"""
        # Stem
        stem = trimesh.creation.cylinder(radius=0.05, height=1.5)
        stem.apply_translation([0, 0, 0.75])
        
        # Flower center
        center = trimesh.creation.uv_sphere(radius=0.2)
        center.apply_translation([0, 0, 1.5])
        
        # Petals
        petals = []
        for i in range(6):
            angle = i * np.pi / 3
            petal = trimesh.creation.box(extents=[0.6, 0.2, 0.1])
            petal.apply_translation([0.4 * np.cos(angle), 0.4 * np.sin(angle), 1.5])
            petals.append(petal)
        
        return trimesh.util.concatenate([stem, center] + petals)
    
    def create_fractal_shape(self) -> trimesh.Trimesh:
        """Create an interesting fractal-inspired shape"""
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
        vertices = mesh.vertices.copy()
        
        # Apply noise to create organic shape
        for i, vertex in enumerate(vertices):
            noise = 0.15 * np.sin(3 * vertex[0]) * np.cos(3 * vertex[1]) * np.sin(3 * vertex[2])
            vertices[i] = vertex * (1 + noise)
        
        mesh.vertices = vertices
        return mesh
    
    async def generate_3d_model(self, prompt: str) -> tuple[trimesh.Trimesh, str, float]:
        """Generate 3D model from text prompt asynchronously"""
        start_time = time.time()
        
        try:
            # Enhance prompt using LangChain + Groq
            enhanced_prompt = await self.enhance_prompt_with_llm(prompt)
            
            # Generate mesh in thread pool
            loop = asyncio.get_event_loop()
            if self.fallback_mode:
                logger.info("Using parametric 3D generation...")
                mesh = await loop.run_in_executor(None, self.generate_parametric_3d, enhanced_prompt)
            else:
                logger.info("Generating with Shap-E...")
                try:
                    mesh = await loop.run_in_executor(None, self._generate_with_shap_e, enhanced_prompt)
                except Exception as e:
                    logger.warning(f"Shap-E failed, using parametric: {e}")
                    mesh = await loop.run_in_executor(None, self.generate_parametric_3d, enhanced_prompt)
            
            # Ensure mesh is valid
            mesh.fix_normals()
            mesh.remove_degenerate_faces()
            
            # Add some color variation based on prompt
            if hasattr(mesh.visual, 'vertex_colors'):
                colors = self.generate_colors_from_prompt(prompt, len(mesh.vertices))
                mesh.visual.vertex_colors = colors
            
            generation_time = time.time() - start_time
            
            # Log generation
            self.generation_history.append({
                'prompt': prompt,
                'enhanced_prompt': enhanced_prompt,
                'timestamp': datetime.now().isoformat(),
                'generation_time': generation_time,
                'vertices': len(mesh.vertices),
                'faces': len(mesh.faces)
            })
            
            return mesh, enhanced_prompt, generation_time
            
        except Exception as e:
            logger.error(f"Error generating 3D model: {e}")
            # Fallback to simple cube
            mesh = trimesh.creation.box(extents=[1, 1, 1])
            return mesh, prompt, 0
    
    def generate_colors_from_prompt(self, prompt: str, num_vertices: int) -> np.ndarray:
        """Generate colors based on the prompt content"""
        prompt_lower = prompt.lower()
        
        # Default color
        base_color = [100, 100, 100, 255]  # Gray
        
        # Color mapping based on keywords
        if any(word in prompt_lower for word in ['red', 'fire', 'rose']):
            base_color = [200, 50, 50, 255]
        elif any(word in prompt_lower for word in ['blue', 'water', 'sky']):
            base_color = [50, 100, 200, 255]
        elif any(word in prompt_lower for word in ['green', 'tree', 'plant']):
            base_color = [50, 150, 50, 255]
        elif any(word in prompt_lower for word in ['yellow', 'gold', 'sun']):
            base_color = [200, 200, 50, 255]
        elif any(word in prompt_lower for word in ['purple', 'violet']):
            base_color = [150, 50, 200, 255]
        elif any(word in prompt_lower for word in ['orange']):
            base_color = [200, 100, 50, 255]
        elif any(word in prompt_lower for word in ['white', 'snow']):
            base_color = [200, 200, 200, 255]
        elif any(word in prompt_lower for word in ['black', 'dark']):
            base_color = [50, 50, 50, 255]
        
        # Add some variation
        colors = np.array([base_color] * num_vertices, dtype=np.uint8)
        variation = np.random.randint(-30, 30, (num_vertices, 3))
        colors[:, :3] = np.clip(colors[:, :3] + variation, 0, 255)
        
        return colors
    
    def _generate_with_shap_e(self, enhanced_prompt: str) -> trimesh.Trimesh:
        """Generate with Shap-E (synchronous)"""
        if not hasattr(self, 'pipe'):
            raise Exception("Shap-E model not loaded")
            
        images = self.pipe(
            enhanced_prompt,
            guidance_scale=15.0,
            num_inference_steps=64,
            frame_size=64,
        ).images
        return self.images_to_mesh(images)
    
    def images_to_mesh(self, images) -> trimesh.Trimesh:
        """Convert Shap-E images to mesh - simplified fallback"""
        # This is a placeholder - actual Shap-E to mesh conversion would be more complex
        return trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    
    async def mesh_to_file(self, mesh: trimesh.Trimesh, format_type: str) -> str:
        """Convert mesh to file format asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._mesh_to_file_sync, mesh, format_type)
    
    def _mesh_to_file_sync(self, mesh: trimesh.Trimesh, format_type: str) -> str:
        """Convert mesh to file format (synchronous)"""
        suffix = f'.{format_type}'
        
        # Create outputs directory if it doesn't exist
        os.makedirs('outputs', exist_ok=True)
        
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir='outputs') as tmp:
            mesh.export(tmp.name, file_type=format_type)
            return tmp.name
    
    async def get_mesh_preview_image(self, mesh: trimesh.Trimesh) -> Optional[str]:
        """Generate a preview image of the mesh asynchronously"""
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._get_mesh_preview_sync, mesh)
        except Exception as e:
            logger.error(f"Error creating preview: {e}")
            return None
    
    def _get_mesh_preview_sync(self, mesh: trimesh.Trimesh) -> str:
        """Generate preview image (synchronous)"""
        try:
            scene = trimesh.Scene(mesh)
            
            # Set up camera for better view
            bounds = mesh.bounds
            center = mesh.centroid
            scale = np.max(bounds[1] - bounds[0])
            
            # Position camera
            camera_distance = scale * 3
            scene.camera_transform = trimesh.transformations.compose_matrix(
                translate=[center[0], center[1], center[2] + camera_distance]
            )
            
            # Render image
            png_data = scene.save_image(resolution=[400, 400], visible=True)
            return base64.b64encode(png_data).decode('utf-8')
        except Exception as e:
            logger.error(f"Preview generation failed: {e}")
            # Return a placeholder image data
            return ""

# Initialize the generator (without starting async tasks yet)
generator = Advanced3DModelGenerator()

# File storage for downloads
file_storage: Dict[str, str] = {}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Main page"""
    return templates.TemplateResponse("index.html", {"request": {}})

@app.post("/api/generate3d", response_model=GenerationResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def generate3d(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate 3D model API endpoint"""
    try:
        logger.info(f"Generating 3D model for prompt: {request.prompt}")
        
        # Generate the model
        mesh, enhanced_prompt, generation_time = await generator.generate_3d_model(request.prompt)
        
        # Get preview image
        preview_image = await generator.get_mesh_preview_image(mesh)
        
        # Convert to requested format
        file_path = await generator.mesh_to_file(mesh, request.format)
        filename = os.path.basename(file_path)
        file_url = f"/download/{request.format}/{filename}"
        
        # Store file path for download
        file_storage[filename] = file_path
        
        # Schedule file cleanup after 1 hour
        background_tasks.add_task(cleanup_file, filename, 3600)
        
        return GenerationResponse(
            success=True,
            enhanced_prompt=enhanced_prompt,
            generation_time=round(generation_time, 2),
            vertices=len(mesh.vertices),
            faces=len(mesh.faces),
            file_url=file_url,
            preview_image=preview_image,
            format=request.format
        )
        
    except Exception as e:
        logger.error(f"Error in generate3d: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """API status endpoint"""
    uptime = datetime.now() - generator.app_start_time
    return StatusResponse(
        status="online",
        model_loaded=generator.model_loaded,
        device=generator.device,
        model_name=generator.model_name,
        total_generations=len(generator.generation_history),
        last_generation=generator.generation_history[-1] if generator.generation_history else None,
        version="2.0.0",
        uptime=f"{uptime.days}d {uptime.seconds//3600}h {(uptime.seconds//60)%60}m",
        features=[
            "Text-to-3D generation",
            "LangChain prompt enhancement" if generator.llm else "Basic prompt processing",
            "Groq LLM integration" if generator.llm else "No LLM integration",
            "Multiple output formats (GLB/OBJ)",
            "Real-time preview",
            "Parametric generation",
            "Async processing",
            "Auto-generated API docs",
            "Color generation from prompts"
        ]
    )

@app.get("/api/history", response_model=HistoryResponse)
async def get_history():
    """Get generation history"""
    return HistoryResponse(
        history=generator.generation_history[-10:],
        total=len(generator.generation_history)
    )

@app.get("/download/{format_type}/{filename}")
async def download_file(format_type: str, filename: str):
    """Download generated file"""
    if filename not in file_storage:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_path = file_storage[filename]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File no longer exists")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type='application/octet-stream'
    )

@app.get("/api/examples", response_model=ExamplesResponse)
async def get_examples():
    """Get example prompts"""
    examples = [
        "A futuristic spaceship with sleek metallic surfaces",
        "A medieval castle tower with stone texture",
        "A modern chair with curved wooden design",
        "A fantasy crystal formation with translucent material",
        "A robotic arm with mechanical joints",
        "A stylized tree with organic branching",
        "A geometric sculpture with angular faces",
        "A vintage red car with chrome details",
        "A mountain landscape with rocky terrain",
        "An abstract spiral structure",
        "A blue flower with delicate petals",
        "A simple house with a triangular roof",
        "A green pyramid with smooth surfaces",
        "A golden torus ring",
        "A purple sphere with textured surface"
    ]
    return ExamplesResponse(examples=examples)

async def cleanup_file(filename: str, delay: int):
    """Clean up temporary files after delay"""
    await asyncio.sleep(delay)
    if filename in file_storage:
        try:
            if os.path.exists(file_storage[filename]):
                os.unlink(file_storage[filename])
            del file_storage[filename]
            logger.info(f"Cleaned up file: {filename}")
        except Exception as e:
            logger.error(f"Error cleaning up file {filename}: {e}")

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    
    # Now start model loading since event loop is running
    generator.start_model_loading()
    
    logger.info("🚀 Advanced 3D Model Generation Server Starting...")
    logger.info(f"📊 Model loading: {generator.model_loaded}")
    logger.info(f"💻 Device: {generator.device}")
    logger.info(f"🤖 LLM Available: {'Yes' if generator.llm else 'No'}")
    logger.info("🌐 FastAPI server ready!")
    logger.info("📚 API docs available at: /docs")
    logger.info("📖 ReDoc available at: /redoc")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("🛑 Shutting down server...")
    
    # Cancel model loading task if it's still running
    if generator._model_loading_task and not generator._model_loading_task.done():
        generator._model_loading_task.cancel()
    
    # Clean up all temporary files
    for filename, filepath in file_storage.items():
        try:
            if os.path.exists(filepath):
                os.unlink(filepath)
        except Exception as e:
            logger.error(f"Error cleaning up {filename}: {e}")
    
    logger.info("✅ Shutdown complete")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )