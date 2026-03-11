import React, { useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Grid } from '@react-three/drei';
import SensorNode from './SensorNode';
import BuildingGeometry from './BuildingGeometry';

export default function Building3DViewer({ 
  nodePositions, 
  modelUrl, 
  activeNode,
  onNodeClick 
}) {
  const [hoveredNode, setHoveredNode] = useState(null);

  return (
    <div style={{ width: '100%', height: '100%' }}>
      <Canvas shadows>
        {/* Camera setup */}
        <PerspectiveCamera 
          makeDefault 
          position={[0, 30, 40]}
          fov={75}
        />
        
        {/* Lighting */}
        <ambientLight intensity={0.4} />
        <directionalLight
          position={[10, 20, 10]}
          intensity={0.8}
          castShadow
          shadow-mapSize={[2048, 2048]}
        />
        <hemisphereLight 
          skyColor="#87CEEB" 
          groundColor="#8B7355" 
          intensity={0.3} 
        />

        {/* Building */}
        <BuildingGeometry modelUrl={modelUrl} />

        {/* Grid helper for spatial reference */}
        <Grid 
          args={[100, 100]}       // was [50, 50]
          cellSize={6} 
          cellColor="#6c6c6c"
          sectionColor="#4a4a4a"
          fadeDistance={100}      // was 60
          position={[0, 0, 0]}
        />

        {/* Sensor nodes */}
        {Object.entries(nodePositions).map(([nodeId, data]) => (
          <SensorNode
            key={nodeId}
            nodeId={nodeId}
            position={[data.x, data.y, data.z]}
            isActive={activeNode === nodeId}
            isHovered={hoveredNode === nodeId}
            sensorTypes={data.sensor_types}
            room={data.room}
            onHover={setHoveredNode}
            onClick={() => onNodeClick(nodeId)}
          />
        ))}

        {/* Camera controls */}
        <OrbitControls 
          enableDamping 
          dampingFactor={0.05}
          minDistance={5}
          maxDistance={500}
          maxPolarAngle={Math.PI / 2.1}
          target={[0, 8, 0]}   // look slightly above ground at building center
        />
      </Canvas>

      {/* Tooltip for hovered node */}
      {hoveredNode && nodePositions[hoveredNode] && (
        <div style={{
          position: 'absolute',
          top: 20,
          left: 20,
          background: 'rgba(0, 0, 0, 0.8)',
          color: 'white',
          padding: '10px 15px',
          borderRadius: '8px',
          pointerEvents: 'none',
          fontFamily: 'sans-serif',
          fontSize: '14px'
        }}>
          <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>
            {hoveredNode}
          </div>
          <div>{nodePositions[hoveredNode].room}</div>
          <div style={{ fontSize: '12px', marginTop: '5px', opacity: 0.8 }}>
            {nodePositions[hoveredNode].sensor_types.join(', ')}
          </div>
        </div>
      )}
    </div>
  );
}