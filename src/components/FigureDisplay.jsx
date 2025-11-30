import React from "react";

export const FigureDisplay = ({ imgBase64, style = style }) => {
  if (!imgBase64) return null; // nothing to show yet

  return (
    <div className="flex justify-center my-4" style={style}>
      <img
        src={`data:image/png;base64,${imgBase64}`}
        alt="Generated Figure"
        className="rounded-lg shadow-lg"
      />
    </div>
  );
};

// export default FigureDisplay;
