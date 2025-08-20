# FPL UI Debugging Guide

## 🔧 Fixed Issues & Solutions

### ✅ **What I've Fixed:**

1. **Added Manual Force Retrain Button** 🔴
   - Emergency failsafe retrain button with confirmation dialog
   - Endpoint: `POST /force-retrain`
   - Completely retrains model and clears all caches

2. **Fixed JavaScript Functionality** 🛠️
   - Removed Alpine.js dependencies that were causing issues
   - Converted to simple `onclick` handlers
   - Added proper error handling and console logging
   - Added loading state management

3. **Enhanced Button Functionality** 📲
   - All Quick Action buttons now use `onclick` instead of Alpine.js
   - Added global loading state to prevent multiple clicks
   - Improved error messages and user feedback

## 🚀 **How to Test the UI**

### 1. **Start the Server**
```bash
# Navigate to your project directory
cd /Users/gabriel/Code/gabriel/fpl/FPL

# Activate virtual environment  
source venv/bin/activate

# Run the web UI
python -m uvicorn src.web_ui_enhanced:app --host 0.0.0.0 --port 8000 --reload
```

### 2. **Open Browser & Test**
```bash
# Open in browser
http://localhost:8000
```

### 3. **Test Each Button (in order)**

#### **Test API Connection First:**
1. **Open Browser Console** (F12 → Console tab)
2. **Look for the 4th status card** ("Test API") and click it
3. **Should see:** `✅ API is working! Status: ok`
4. **If this fails:** API server isn't running

#### **Test Quick Actions:**
1. **📝 Log Current Suggestion** - Should show success dialog
2. **📊 Collect Last Results** - May show "no data" initially (normal)  
3. **📈 Generate Report** - Should show report summary
4. **🔧 Auto Improve** - Should check model status
5. **🔴 FORCE RETRAIN** - Emergency option (confirmation required)

#### **Test Cache Panel:**
1. **Click "Show Cache"** - Should expand panel
2. **Click "Refresh"** - Should load cache status
3. **Check Console** - Should see cache data logged

#### **Test Tab Functions:**
1. **Accuracy Analytics Tab** → Click "Generate Report"
2. **Team History Tab** → Click "Load History"  
3. **Model Status Tab** → Click "Check Status"

## 🐛 **If Buttons Still Don't Work:**

### **Check Browser Console:**
1. Open F12 → Console tab
2. Look for any red error messages
3. Common issues:
   - `fetch is not defined` → Browser compatibility issue
   - `404 errors` → API endpoints not found  
   - `CORS errors` → Cross-origin issues
   - `NetworkError` → Server not running

### **Manual API Tests:**
```bash
# Test API directly
curl http://localhost:8000/health
curl -X POST http://localhost:8000/quick-actions/log-current-suggestion
curl http://localhost:8000/cache-status
```

### **Check Server Logs:**
Look at the uvicorn console output for:
- `404 Not Found` errors
- Python exceptions  
- Missing imports

### **Common Import Issues:**
If you see import errors, the new files might need importing:
```python
# In src/web_ui_enhanced.py, check these imports exist:
from .comprehensive_accuracy_tracker import ComprehensiveAccuracyTracker
from .cache_manager import CacheManager
```

## 🎯 **Smart Learning Mode (Your Choice)**

**Current Setting:** Smart Mode (recommended)
- ✅ Only retrains when performance degrades
- ✅ Efficient resource usage  
- ✅ Prevents overfitting
- ✅ Manual override available (Force Retrain button)

**How It Works:**
1. **Monitors prediction accuracy** each gameweek
2. **If accuracy drops** → Triggers retraining
3. **If accuracy good** → Just updates features
4. **Manual override** → Red Force Retrain button

## 🔴 **Emergency Force Retrain**

**When to Use:**
- Model seems stuck or not learning
- After major data issues
- Suspect model corruption
- Want to start fresh

**What It Does:**
- Completely retrains from scratch
- Uses ALL available data
- Clears ALL caches
- Takes several minutes

**How to Use:**
1. Click red "FORCE RETRAIN" button
2. Read warning dialog carefully
3. Click "OK" only if you're sure
4. Wait for completion message

## 📱 **Expected Behavior**

### **Working Buttons Should:**
- Show loading/processing states
- Display success/error messages  
- Log activity to browser console
- Provide user feedback

### **Data Flow:**
1. **Log Suggestion** → Saves team for tracking
2. **Collect Results** → Gathers actual gameweek data
3. **Generate Report** → Analyzes accuracy trends
4. **Auto Improve** → Smart retraining when needed
5. **Force Retrain** → Emergency full retrain

## 🚨 **If Nothing Works**

### **Fallback Options:**
```bash
# 1. Try basic predictions endpoint directly
curl http://localhost:8000/predictions

# 2. Check if any endpoints work
curl http://localhost:8000/health

# 3. Restart server with verbose logging
python -m uvicorn src.web_ui_enhanced:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

### **Nuclear Option - Simplified UI:**
If all else fails, I can create a minimal HTML page with just working buttons for testing.

---

**The UI should now work properly with simplified JavaScript and better error handling. Try the steps above and let me know what you see in the browser console!**