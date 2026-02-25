import React, { Suspense } from 'react';
import { useGLTF } from '@react-three/drei';
import * as THREE from 'three';

function LoadedModel({ modelUrl }) {
  const { scene } = useGLTF(modelUrl);

  const box = new THREE.Box3().setFromObject(scene);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());

  const scale = 15 / size.y;

  // Log so we can see what's happening
  console.log('size:', size);
  console.log('center:', center);
  console.log('box.min:', box.min);
  console.log('box.max:', box.max);
  console.log('scale:', scale);
  console.log('final position:', {
    x: -center.x * scale,
    y: -box.min.y * scale,
    z: -center.z * scale
  });

  return (
    <group
      scale={[scale, scale, scale]}
      position={[
        -center.x * scale,  // center on X
        -box.min.y * scale, // lift so bottom sits at y=0
        -center.z * scale   // center on Z
      ]}
    >
      <primitive object={scene} />
    </group>
  );
}

function ProceduralBuilding() {
  const buildingWidth = 40;
  const buildingDepth = 20;
  const floorHeight = 4;
  const numFloors = 3;

  return (
    <group>
      <mesh position={[0, (numFloors * floorHeight) / 2, 0]} receiveShadow>
        <boxGeometry args={[buildingWidth, numFloors * floorHeight, buildingDepth]} />
        <meshStandardMaterial 
          color="#d4d4d4" 
          transparent 
          opacity={0.6}
          side={THREE.DoubleSide}
        />
      </mesh>

      {[1, 2, 3].map((floor) => (
        <mesh 
          key={floor} 
          position={[0, floor * floorHeight, 0]}
        >
          <boxGeometry args={[buildingWidth + 0.2, 0.1, buildingDepth + 0.2]} />
          <meshStandardMaterial color="#888888" />
        </mesh>
      ))}
    </group>
  );
}

export default function BuildingGeometry({ modelUrl }) {
  if (modelUrl) {
    // Preload so the model is cached and survives rerenders/remounts
    useGLTF.preload(modelUrl);
    return (
      <Suspense fallback={null}>
        <LoadedModel modelUrl={modelUrl} />
      </Suspense>
    );
  }

  return <ProceduralBuilding />;
}