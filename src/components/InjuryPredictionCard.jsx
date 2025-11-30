import React from 'react';

export const InjuryPredictionCard = ({ label = "Injury Prediction", status = "Injured", percentage = 80, opacity=opacity }) => {

  const color = status === "Injured" ? "#ef4444" : "#22c55e"; // red for injured, green for non-injured

  return (
    <div className="bg-white rounded-lg shadow-sm border p-6" style={{ width: "100%" }}>
      <h2 className="text-lg font-semibold mb-4">{label}</h2>
      
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-700" style={{ opacity: opacity}}>{status}</span>  
        <span className="text-sm font-medium text-gray-700" style={{ opacity: opacity}}>{percentage}%</span>
      </div>

      {/* Progress Bar */}
      <div className="w-full bg-gray-200 rounded-full h-4">
        <div
          className="h-4 rounded-full"
          style={{
            width: `${percentage}%`,
            backgroundColor: color,
            transition: "width 0.5s ease-in-out",
            opacity: opacity
          }}
        />
      </div>

      {/* Optional visual icon */}
      <div className="mt-4 flex justify-center" style={{ opacity: opacity}}>
        {status === "Injured" ? (
          <svg className="w-8 h-8 text-red-500 animate-pulse" fill="currentColor" viewBox="0 0 20 20">
            <path d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-11v4H9V7h2zm0 6v2H9v-2h2z" />
          </svg>
        ) : (
          <svg className="w-8 h-8 text-green-500 animate-bounce" fill="currentColor" viewBox="0 0 20 20">
            <path d="M10 18a8 8 0 100-16 8 8 0 000 16zm-1-5l-3-3 1.414-1.414L9 10.172l4.586-4.586L15 7l-6 6z" />
          </svg>
        )}
      </div>
    </div>
  );
};
