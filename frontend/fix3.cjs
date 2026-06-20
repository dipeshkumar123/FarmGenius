const fs = require('fs');

function fixFile(filePath) {
    if (!fs.existsSync(filePath)) return;
    let content = fs.readFileSync(filePath, 'utf8');
    
    // Remove all ease definitions in transition objects
    content = content.replace(/ease:\s*\[[\d.,\s]+\]\s*,?/g, '');
    content = content.replace(/ease:\s*['"][a-zA-Z]+['"]\s*,?/g, '');
    
    // Fix Plant import
    content = content.replace(/Plant,/g, 'Leaf as Plant,');
    
    fs.writeFileSync(filePath, content);
}

const pages = [
    'src/pages/DashboardPage.tsx',
    'src/pages/LandingPage.tsx',
    'src/pages/LoginPage.tsx',
    'src/pages/ScanPage.tsx',
    'src/pages/MarketPage.tsx'
];

pages.forEach(fixFile);

console.log("Applied final fixes.");
