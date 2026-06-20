import React, { useState } from 'react';
import { PageWrapper, AnimatedSection } from '../components/ui/PageWrapper';
import { Leaf, MapPin, Drop, Info, ChatCircleText, CaretDown, CheckCircle } from 'phosphor-react';
import { motion, AnimatePresence } from 'framer-motion';

const mockCrops = [
  { rank: 1, name: 'Paddy (Rice)', emoji: '🌾', suitability: 0.96, expectedYield: '45 q/acre', marketPrice: '₹2,060/q', profitEstimate: '₹92,700', season: 'Kharif', water: 'High', duration: '120-150 days' },
  { rank: 2, name: 'Maize', emoji: '🌽', suitability: 0.82, expectedYield: '32 q/acre', marketPrice: '₹1,920/q', profitEstimate: '₹61,440', season: 'Kharif', water: 'Medium', duration: '90-110 days' },
  { rank: 3, name: 'Soybean', emoji: '🫘', suitability: 0.74, expectedYield: '18 q/acre', marketPrice: '₹4,420/q', profitEstimate: '₹79,560', season: 'Kharif', water: 'Low', duration: '90-100 days' },
];

export default function CropsPage() {
  const [step, setStep] = useState<1 | 2>(1);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [expandedCrop, setExpandedCrop] = useState<number | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    setTimeout(() => {
      setIsSubmitting(false);
      setStep(2);
    }, 1500);
  };

  const resetForm = () => setStep(1);

  return (
    <PageWrapper className="p-4 pb-24 bg-[#F1F8E9]">
      <AnimatedSection custom={0} className="mb-6">
        <h1 className="text-2xl font-poppins font-bold text-[#1B2B1D] mb-1">Crop Recommendation</h1>
        <p className="text-sm text-[#546E7A]">Find the best crops for your farm based on soil and weather.</p>
      </AnimatedSection>

      <AnimatePresence mode="wait">
        {step === 1 ? (
          <motion.form 
            key="form"
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            onSubmit={handleSubmit} 
            className="space-y-4"
          >
            <div className="bg-white p-5 rounded-2xl shadow-sm border border-[#E8F5E9]">
              <div className="space-y-4">
                {/* Location */}
                <div>
                  <label className="block text-sm font-medium text-[#1B2B1D] mb-1">Farm Location</label>
                  <div className="flex items-center bg-[#F1F8E9] p-3 rounded-xl border border-transparent focus-within:border-[#2E7D32]">
                    <MapPin className="text-[#2E7D32] mr-2" weight="fill" />
                    <input type="text" defaultValue="Dharwad, Karnataka" className="bg-transparent border-none outline-none w-full text-sm text-[#1B2B1D]" />
                  </div>
                </div>

                {/* Soil Type */}
                <div>
                  <label className="block text-sm font-medium text-[#1B2B1D] mb-1">Soil Type</label>
                  <div className="relative">
                    <select className="w-full bg-[#F1F8E9] p-3 rounded-xl appearance-none outline-none text-sm text-[#1B2B1D] focus:border-[#2E7D32] border border-transparent">
                      <option>Loamy</option>
                      <option>Sandy</option>
                      <option>Clay</option>
                      <option>Black</option>
                      <option>Red</option>
                      <option>Alluvial</option>
                    </select>
                    <CaretDown className="absolute right-3 top-3.5 text-[#546E7A] pointer-events-none" />
                  </div>
                </div>

                {/* Water Availability */}
                <div>
                  <label className="block text-sm font-medium text-[#1B2B1D] mb-2">Water Availability</label>
                  <div className="flex space-x-2">
                    {['Rainfed', 'Irrigated', 'Drip'].map((w) => (
                      <label key={w} className="flex-1">
                        <input type="radio" name="water" className="peer sr-only" defaultChecked={w === 'Rainfed'} />
                        <div className="text-center py-2 px-1 text-sm bg-[#F1F8E9] text-[#546E7A] rounded-xl cursor-pointer peer-checked:bg-[#2E7D32] peer-checked:text-white transition-colors">
                          {w}
                        </div>
                      </label>
                    ))}
                  </div>
                </div>

                {/* Farm Size */}
                <div>
                  <label className="block text-sm font-medium text-[#1B2B1D] mb-1">Farm Size</label>
                  <div className="flex space-x-2">
                    <input type="number" defaultValue={2} className="flex-1 bg-[#F1F8E9] p-3 rounded-xl outline-none text-sm text-[#1B2B1D] focus:border-[#2E7D32] border border-transparent" />
                    <div className="relative w-1/3">
                      <select className="w-full bg-[#F1F8E9] p-3 rounded-xl appearance-none outline-none text-sm text-[#1B2B1D] focus:border-[#2E7D32] border border-transparent">
                        <option>Acres</option>
                        <option>Hectares</option>
                      </select>
                      <CaretDown className="absolute right-3 top-3.5 text-[#546E7A] pointer-events-none" />
                    </div>
                  </div>
                </div>

                {/* Season */}
                <div>
                  <label className="block text-sm font-medium text-[#1B2B1D] mb-2">Target Season</label>
                  <div className="flex space-x-2">
                    {['Kharif', 'Rabi', 'Zaid'].map((s) => (
                      <label key={s} className="flex-1">
                        <input type="radio" name="season" className="peer sr-only" defaultChecked={s === 'Kharif'} />
                        <div className="text-center py-2 px-1 text-sm bg-[#F1F8E9] text-[#546E7A] rounded-xl cursor-pointer peer-checked:bg-[#F9A825] peer-checked:text-white transition-colors font-medium">
                          {s}
                        </div>
                      </label>
                    ))}
                  </div>
                </div>

                {/* NPK Values */}
                <div>
                  <label className="block text-sm font-medium text-[#1B2B1D] mb-1">Soil NPK Values <span className="text-[#546E7A] font-normal text-xs">(Optional)</span></label>
                  <div className="flex space-x-3">
                    {['N', 'P', 'K'].map((npk) => (
                      <div key={npk} className="flex-1 relative">
                        <span className="absolute left-3 top-3.5 text-[#546E7A] text-xs font-bold">{npk}</span>
                        <input type="number" placeholder="--" className="w-full bg-[#F1F8E9] p-3 pl-8 rounded-xl outline-none text-sm text-[#1B2B1D] focus:border-[#2E7D32] border border-transparent" />
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            <button 
              type="submit" 
              disabled={isSubmitting}
              className="w-full bg-[#2E7D32] text-white py-4 rounded-xl font-semibold text-lg flex items-center justify-center space-x-2 hover:bg-[#1B5E20] active:scale-[0.98] transition-all disabled:opacity-70 shadow-md shadow-[#2E7D32]/30"
            >
              {isSubmitting ? (
                <div className="flex items-center space-x-2">
                  <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Analyzing Data...</span>
                </div>
              ) : (
                <>
                  <Leaf weight="fill" />
                  <span>Get AI Recommendations →</span>
                </>
              )}
            </button>
          </motion.form>
        ) : (
          <motion.div 
            key="results"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="space-y-4"
          >
            <div className="flex justify-between items-center mb-2">
              <h2 className="text-lg font-semibold text-[#1B2B1D]">Top Recommendations</h2>
              <button onClick={resetForm} className="text-sm text-[#2E7D32] font-medium hover:underline">Edit Details</button>
            </div>

            {mockCrops.map((crop, idx) => {
              const isExpanded = expandedCrop === idx;
              const badgeColors = ['bg-yellow-400', 'bg-slate-300', 'bg-amber-600'];
              const badgeColor = badgeColors[idx] || 'bg-gray-200';
              
              return (
                <motion.div 
                  key={idx}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: idx * 0.1 }}
                  className={`bg-white rounded-2xl border-2 overflow-hidden shadow-sm transition-all ${idx === 0 ? 'border-[#F9A825]' : 'border-transparent'}`}
                >
                  <div className="p-4 cursor-pointer" onClick={() => setExpandedCrop(isExpanded ? null : idx)}>
                    <div className="flex justify-between items-start mb-3">
                      <div className="flex items-center space-x-3">
                        <div className="text-4xl bg-[#F1F8E9] w-14 h-14 rounded-full flex items-center justify-center shadow-inner">
                          {crop.emoji}
                        </div>
                        <div>
                          <div className="flex items-center space-x-2">
                            <h3 className="font-poppins font-bold text-lg text-[#1B2B1D]">{crop.name}</h3>
                            {idx < 3 && <span className={`text-[10px] uppercase font-bold text-white px-2 py-0.5 rounded-full ${badgeColor}`}>#{crop.rank} Match</span>}
                          </div>
                          <div className="text-sm text-[#546E7A] flex items-center mt-0.5">
                            <span className="font-medium text-[#2E7D32]">{Math.round(crop.suitability * 100)}% Suitability</span>
                          </div>
                        </div>
                      </div>
                      <CaretDown className={`text-[#546E7A] transition-transform ${isExpanded ? 'rotate-180' : ''}`} />
                    </div>

                    <div className="w-full bg-[#E8F5E9] rounded-full h-2 mb-4 overflow-hidden">
                      <div className="bg-[#2E7D32] h-full rounded-full transition-all duration-1000" style={{ width: `${crop.suitability * 100}%` }}></div>
                    </div>

                    <div className="grid grid-cols-2 gap-3 mb-2">
                      <div className="bg-[#F1F8E9] p-2.5 rounded-xl">
                        <div className="text-xs text-[#546E7A] mb-0.5">Expected Yield</div>
                        <div className="font-semibold text-[#1B2B1D]">{crop.expectedYield}</div>
                      </div>
                      <div className="bg-yellow-50 p-2.5 rounded-xl border border-yellow-100">
                        <div className="text-xs text-[#546E7A] mb-0.5">Est. Profit</div>
                        <div className="font-semibold text-[#EF6C00]">{crop.profitEstimate}</div>
                      </div>
                    </div>

                    <AnimatePresence>
                      {isExpanded && (
                        <motion.div 
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: 'auto' }}
                          exit={{ opacity: 0, height: 0 }}
                          className="pt-3 border-t border-gray-100 mt-3"
                        >
                          <div className="grid grid-cols-2 gap-y-2 gap-x-4 text-sm mb-4">
                            <div className="flex items-center text-[#546E7A]"><CheckCircle size={16} className="text-[#2E7D32] mr-1.5"/> Season: <span className="text-[#1B2B1D] font-medium ml-1">{crop.season}</span></div>
                            <div className="flex items-center text-[#546E7A]"><Drop size={16} className="text-blue-500 mr-1.5"/> Water: <span className="text-[#1B2B1D] font-medium ml-1">{crop.water}</span></div>
                            <div className="flex items-center text-[#546E7A]"><Info size={16} className="text-[#F9A825] mr-1.5"/> Market: <span className="text-[#1B2B1D] font-medium ml-1">{crop.marketPrice}</span></div>
                            <div className="flex items-center text-[#546E7A]"><CaretDown size={16} className="text-gray-400 mr-1.5 transform -rotate-90"/> Duration: <span className="text-[#1B2B1D] font-medium ml-1">{crop.duration}</span></div>
                          </div>
                          
                          <button className="w-full py-2.5 bg-[#E8F5E9] text-[#2E7D32] rounded-xl font-medium text-sm flex items-center justify-center hover:bg-[#2E7D32] hover:text-white transition-colors active:scale-95">
                            <ChatCircleText weight="fill" className="mr-2" size={18} />
                            Ask AI about {crop.name}
                          </button>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </motion.div>
              );
            })}
          </motion.div>
        )}
      </AnimatePresence>
    </PageWrapper>
  );
}
