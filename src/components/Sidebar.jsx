import React from 'react';

export const Sidebar = ({ selectedPatient, setSelectedPatient, selectedTask, setSelectedTask, onAnalyze, loading }) => {
  
    const handleChange = (e) => {
      setSelectedPatient(e.target.value);
      console.log("Selected patient:", e.target.value);
    };


  return (
    <div className="w-72 bg-white border-r border-gray-200 flex flex-col">
      <div className="p-4 border-b border-gray-200">
        <h1 className="text-xl font-bold text-gray-800">XAI Healthcare</h1>
        <p className="text-xs text-gray-500 mt-1">Motion Assessment Platform</p>
      </div>

      <div className="p-4 space-y-4">
        <div>
          <label className="text-xs font-medium text-gray-700 block mb-2">Select Patient</label>
          <select 
            value={selectedPatient}
            onChange={handleChange}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="" disabled>
              Select Patient
            </option>
            <option>patient_1</option>
            <option>patient_2</option>
            <option>patient_3</option>
            <option>patient_4</option>
            <option>patient_5</option>
            <option>patient_6</option>
            <option>control_1</option>
            <option>control_2</option>
            <option>control_3</option>
            <option>control_4</option>
            <option>control_5</option>
            <option>control_6</option>
          </select>
        </div>

        <div>
          <label className="text-xs font-medium text-gray-700 block mb-2">Select Task</label>
          <select 
            value={selectedTask}
            onChange={(e) => setSelectedTask(e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="" disabled>
              Select Task
            </option>
            <option>Jar opening</option>
            <option>Key turning</option>
            <option>Wall cleaning</option>
            <option>Backwashing</option>
            <option>Knife slicing</option>
            <option>Hammering</option>
          </select>
        </div>

        <button 
          onClick={onAnalyze}
          // className="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 rounded-lg transition-colors"
          className={`w-full bg-blue-600 text-white font-medium py-3 rounded-lg transition-colors 
                  ${loading ? 'opacity-50 cursor-not-allowed' : 'hover:bg-blue-700'}`}
          // disabled={loading}
        >
          {loading ? 'Analyzing...' : 'Go'} 
        </button>
      </div>
    </div>
  );
};