const fs = require('fs');
const path = require('path');

function walkDir(dir, callback) {
  fs.readdirSync(dir).forEach(f => {
    let dirPath = path.join(dir, f);
    let isDirectory = fs.statSync(dirPath).isDirectory();
    isDirectory ? walkDir(dirPath, callback) : callback(path.join(dir, f));
  });
}

walkDir('src', function(filePath) {
  if (filePath.endsWith('.tsx')) {
    let content = fs.readFileSync(filePath, 'utf8');
    let original = content;

    // Remove ease
    content = content.replace(/ease:\s*(?:'[^']+'|"[^"]+"|\[[\d.,\s]+\])\s*,?/g, '');
    
    // Fix imports
    content = content.replace(/import\s*\{([^}]*)KeyboardEvent([^}]*)\}\s*from\s*'react';/g, "import { $1 $2 } from 'react';\nimport type { KeyboardEvent } from 'react';");
    content = content.replace(/import\s*\{([^}]*)ChangeEvent([^}]*)\}\s*from\s*'react';/g, "import { $1 $2 } from 'react';\nimport type { ChangeEvent } from 'react';");
    
    // Clean up empty imports
    content = content.replace(/import\s*\{\s*,\s*\}\s*from\s*'react';\n/g, '');
    content = content.replace(/import\s*\{\s*\}\s*from\s*'react';\n/g, '');
    
    // Fix Microscope
    content = content.replace(/Microscope,/g, 'MagnifyingGlass as Microscope,');

    if (content !== original) {
      fs.writeFileSync(filePath, content);
    }
  }
});
console.log("Fixes applied to all tsx files.");
