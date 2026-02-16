import React from 'react';
import ReactDOM from 'react-dom';
import { Streamlit, withStreamlitConnection } from 'streamlit-component-lib';
import Building3DViewer from './Building3DViewer';

function StreamlitComponent(props) {
  const { nodePositions, modelUrl, activeNode, height } = props.args;

  const handleNodeClick = (nodeId) => {
    Streamlit.setComponentValue(nodeId);
  };

  // Set initial height
  React.useEffect(() => {
    Streamlit.setFrameHeight(height || 600);
  }, [height]);

  return (
    <div style={{ width: '100%', height: `${height || 600}px` }}>
      <Building3DViewer
        nodePositions={nodePositions || {}}
        modelUrl={modelUrl}
        activeNode={activeNode}
        onNodeClick={handleNodeClick}
      />
    </div>
  );
}

const StreamlitBuildingViewer = withStreamlitConnection(StreamlitComponent);

ReactDOM.render(
  <React.StrictMode>
    <StreamlitBuildingViewer />
  </React.StrictMode>,
  document.getElementById('root')
);