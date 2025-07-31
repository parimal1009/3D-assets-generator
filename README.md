# ⚡ Ultra-Fast 3D Model Generator

<div align="center">

![3D Generator UI](https://github.com/parimal1009/3D-assets-generator/blob/main/images/Screenshot%202025-07-27%20123326.png?raw=true)

**Lightning-Speed AI-Powered 3D Asset Creation - Optimized for Speed & Quality**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/Version-3.0.0-purple.svg)]()

</div>

## 🎯 Overview

Ultra-Fast 3D Model Generator is a cutting-edge AI-powered application that creates high-quality 3D models from text descriptions in under 10 seconds. Built with FastAPI and advanced machine learning techniques, it offers lightning-fast generation with intelligent prompt enhancement and multiple output formats.

## ✨ Key Features

- ⚡ **Ultra-Fast Generation**: Create 3D models in under 10 seconds
- 🧠 **AI Prompt Enhancement**: Intelligent prompt processing with Groq LLM integration
- 🎨 **Multiple Formats**: Support for GLB and OBJ file formats
- 🖼️ **Instant Previews**: Real-time 3D model previews
- 🚀 **Smart Caching**: Advanced caching system for repeated prompts
- 📊 **Quality Scoring**: Automatic quality assessment of generated models
- 🎯 **Parametric Generation**: Optimized algorithms for various object types
- 🔄 **Async Processing**: Non-blocking background generation
- 🧹 **Auto-Cleanup**: Automatic file management and cleanup
- 📱 **Responsive UI**: Modern, mobile-friendly interface

## 🖼️ Sample Outputs

<div align="center">

### Sample 3D Asset 1
![Sample 1](https://github.com/parimal1009/3D-assets-generator/blob/main/images/Screenshot%202025-07-31%20213405.png?raw=true)

### Sample 3D Asset 2
![Sample 2](https://github.com/parimal1009/3D-assets-generator/blob/main/images/Screenshot%202025-07-27%20123720.png?raw=true)

### Sample 3D Asset 3
![Sample 3](https://github.com/parimal1009/3D-assets-generator/blob/main/images/Screenshot%202025-07-31%20213208.png?raw=true)

</div>

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for enhanced performance)
- Groq API key (optional, for AI prompt enhancement)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/parimal1009/3D-assets-generator.git
   cd 3D-assets-generator
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables** (optional)
   ```bash
   # Create .env file
   echo "GROQ_API_KEY=your_groq_api_key_here" > .env
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

5. **Access the application**
   - Open your browser and go to `http://localhost:8000`
   - API documentation available at `http://localhost:8000/docs`

## 🎮 Usage

### Web Interface

1. **Enter a description** of the 3D model you want to create
2. **Select output format** (GLB or OBJ)
3. **Click "Generate 3D Model"**
4. **Download** your generated model

### Example Prompts

- `"A futuristic silver sports car with aerodynamic design"`
- `"A cozy wooden house with red brick chimney"`
- `"A tall oak tree with detailed branches and leaves"`
- `"A delicate pink rose with layered petals"`
- `"A sleek modern chair with curved metal frame"`

### API Usage

```python
import requests

# Generate a 3D model
response = requests.post("http://localhost:8000/api/generate3d", json={
    "prompt": "A red metallic cube with smooth edges",
    "format": "glb"
})

if response.status_code == 200:
    result = response.json()
    print(f"Model generated in {result['generation_time']}s")
    print(f"Download URL: {result['file_url']}")
    print(f"Quality Score: {result['quality_score']}/10")
```

## 🏗️ Architecture

### Core Components

- **FastAPI Backend**: High-performance async web framework
- **Trimesh Engine**: Advanced 3D mesh processing and generation
- **Groq LLM Integration**: AI-powered prompt enhancement
- **Optimized Caching**: Smart caching for repeated requests
- **Background Workers**: Multi-threaded processing system

### Technology Stack

- **Backend**: FastAPI, Uvicorn, Pydantic
- **3D Processing**: Trimesh, NumPy, PyTorch
- **AI Integration**: LangChain, Groq API
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **File Handling**: Async file operations, automatic cleanup

## 📊 Performance Metrics

- **Average Generation Time**: < 10 seconds
- **Cache Hit Rate**: Up to 60% for repeated prompts
- **Supported Formats**: GLB, OBJ
- **Quality Score Range**: 0-10 (automatic assessment)
- **Concurrent Processing**: Multi-threaded generation
- **Memory Optimization**: Efficient mesh handling

## 🔧 Configuration

### Environment Variables

```bash
# Optional: Groq API key for AI prompt enhancement
GROQ_API_KEY=your_groq_api_key_here

# Optional: Custom server settings
HOST=0.0.0.0
PORT=8000
```

### Advanced Settings

The application includes several optimization features:

- **Worker Threads**: Configurable background processing
- **Cache Size**: Adjustable generation cache
- **File Cleanup**: Automatic cleanup after 30 minutes
- **Quality Thresholds**: Customizable quality scoring

## 🎨 Supported Object Types

### Basic Shapes
- Cubes, Spheres, Cylinders
- Pyramids, Torus, Cones
- Capsules, Diamonds

### Complex Objects
- Vehicles (cars, trucks)
- Buildings (houses, structures)
- Nature (trees, flowers)
- Furniture (chairs, tables)

### Custom Shapes
- Abstract forms
- Organic sculptures
- Geometric patterns
- Artistic designs

## 📁 Project Structure

```
3d-asset/
├── main.py                 # Main FastAPI application
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html         # Web interface
├── static/                # Static assets
├── outputs/               # Generated 3D models
└── README.md             # This file
```

## 🔌 API Endpoints

### Core Endpoints

- `GET /` - Web interface
- `POST /api/generate3d` - Generate 3D model
- `GET /api/status` - System status and metrics
- `GET /api/examples` - Example prompts
- `GET /api/history` - Generation history
- `GET /download/{format}/{filename}` - Download generated model

### Response Format

```json
{
  "success": true,
  "enhanced_prompt": "AI-enhanced description",
  "generation_time": 8.5,
  "vertices": 1250,
  "faces": 2480,
  "file_url": "/download/glb/tmp123.glb",
  "preview_image": "base64_encoded_image",
  "format": "glb",
  "quality_score": 8.2
}
```

## 🚀 Deployment

### Local Development

```bash
# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment

```bash
# Production mode
python main.py
```

### Docker Deployment (Optional)

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py"]
```

## 🔍 Troubleshooting

### Common Issues

1. **Slow Generation**
   - Check GPU availability
   - Verify Groq API key (optional)
   - Monitor system resources

2. **Memory Issues**
   - Reduce worker threads
   - Clear generation cache
   - Restart application

3. **File Download Issues**
   - Check file permissions
   - Verify cleanup settings
   - Monitor disk space

### Performance Optimization

- Use CUDA-compatible GPU for faster processing
- Configure appropriate cache sizes
- Monitor system resources
- Optimize prompt descriptions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FastAPI** - Modern web framework for building APIs
- **Trimesh** - 3D mesh processing library
- **Groq** - High-performance AI inference
- **LangChain** - LLM application framework
- **PyTorch** - Machine learning framework

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/parimal1009/3D-assets-generator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/parimal1009/3D-assets-generator/discussions)


## 🔄 Version History

- **v3.0.0** - Ultra-fast generation with advanced caching
- **v2.0.0** - AI prompt enhancement and quality scoring
- **v1.0.0** - Initial release with basic 3D generation

---

<div align="center">



⭐ **Star this repository if you found it helpful!**

</div> 
