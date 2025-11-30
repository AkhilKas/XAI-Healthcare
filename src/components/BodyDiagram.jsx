import React from 'react';

export const BodyDiagram = ({ values = { head: 20, left: 80, right: 50 }, style=style}) => {
  // Helper to map percentage to color
  const getColor = (value) => {
    if (value >= 66) return "#ef4444"; // red
    if (value >= 33) return "#fbbf24"; // yellow
    return "#22c55e"; // green
  };

  return (
    <div className="relative w-full h-64 bg-gray-50 rounded-lg flex items-center justify-center" style={style}>
      <div className="relative">
        <svg width="120" height="200" viewBox="0 0 120 200">
          {/* Head */}
          <ellipse cx="60" cy="20" rx="15" ry="18" fill={getColor(values.head)} />
          
          {/* Torso */}
          <rect x="50" y="35" width="20" height="50" rx="8" fill="#e5e7eb" />
          
          {/* Left Arm */}
          <line
            x1="50"
            y1="45"
            x2="20"
            y2="75"
            stroke={getColor(values.left)}
            strokeWidth="8"
            strokeLinecap="round"
          />
          <circle
            cx="20"
            cy="75"
            r="12"
            fill={getColor(values.left)}
            opacity="0.5"
          />
          
          {/* Right Arm */}
          <line
            x1="70"
            y1="45"
            x2="100"
            y2="75"
            stroke={getColor(values.right)}
            strokeWidth="8"
            strokeLinecap="round"
          />
          <circle
            cx="100"
            cy="75"
            r="12"
            fill={getColor(values.right)}
            opacity="0.5"
          />
          
          {/* Legs */}
          <line x1="55" y1="85" x2="45" y2="140" stroke="#e5e7eb" strokeWidth="8" strokeLinecap="round" />
          <line x1="65" y1="85" x2="75" y2="140" stroke="#e5e7eb" strokeWidth="8" strokeLinecap="round" />
        </svg>

        {/* Left Label */}
        <div className="absolute left-0 top-24 transform -translate-x-full text-sm font-medium text-gray-700">
          Left shoulder
        </div>

        {/* Right Label */}
        <div className="absolute right-0 top-24 transform translate-x-full text-sm font-medium text-gray-700">
          Right shoulder
        </div>
      </div>
    </div>
  );
};
