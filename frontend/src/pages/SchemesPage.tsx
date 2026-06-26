import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { PageWrapper, AnimatedSection } from '../components/ui/PageWrapper';
import { MagnifyingGlass, CheckCircle, ArrowRight } from 'phosphor-react';
import { motion } from 'framer-motion';

const schemes = [
  { id: 1, name: 'PM-KISAN', fullName: 'Pradhan Mantri Kisan Samman Nidhi', category: 'Direct Benefit', amount: '₹6,000/year', eligibility: 'Small & marginal farmers with less than 2 hectares', deadline: 'Ongoing', color: '#FF6B35', emoji: '🏛️', description: 'Direct income support to eligible farmer families in 3 equal installments.', applyLink: 'https://pmkisan.gov.in' },
  { id: 2, name: 'PMFBY', fullName: 'Pradhan Mantri Fasal Bima Yojana', category: 'Crop Insurance', amount: 'Up to full coverage', eligibility: 'All farmers growing notified crops', deadline: 'Before sowing', color: '#2196F3', emoji: '🛡️', description: 'Crop insurance scheme providing financial support if crops fail due to natural calamities.', applyLink: 'https://pmfby.gov.in' },
  { id: 3, name: 'KCC', fullName: 'Kisan Credit Card', category: 'Credit/Loans', amount: 'Up to ₹3 lakh @ 7%', eligibility: 'All farmers, tenant farmers, sharecroppers', deadline: 'Ongoing', color: '#9C27B0', emoji: '💳', description: 'Short-term credit needs for cultivation and allied activities at subsidized interest rates.', applyLink: 'https://www.nabard.org/kisan-credit-card' },
  { id: 4, name: 'PMKSY', fullName: 'PM Krishi Sinchayee Yojana', category: 'Irrigation', amount: '55% subsidy', eligibility: 'Individual farmers, farmer groups', deadline: 'State-wise', color: '#00BCD4', emoji: '💧', description: 'Micro-irrigation subsidy scheme for drip and sprinkler irrigation systems.', applyLink: 'https://pmksy.gov.in' },
  { id: 5, name: 'eNAM', fullName: 'National Agriculture Market', category: 'Market Access', amount: 'Free platform', eligibility: 'Any registered farmer', deadline: 'Ongoing', color: '#4CAF50', emoji: '🏪', description: 'Online trading portal for farmers to sell produce directly to buyers across India.', applyLink: 'https://www.enam.gov.in' },
];

const categories = ['All', 'Direct Benefit', 'Crop Insurance', 'Credit/Loans', 'Irrigation', 'Market Access'];

export default function SchemesPage() {
  const navigate = useNavigate();
  const [search, setSearch] = useState('');
  const [activeCategory, setActiveCategory] = useState('All');

  const filteredSchemes = schemes.filter(s => {
    const matchesSearch = s.name.toLowerCase().includes(search.toLowerCase()) || s.fullName.toLowerCase().includes(search.toLowerCase());
    const matchesCategory = activeCategory === 'All' || s.category === activeCategory;
    return matchesSearch && matchesCategory;
  });

  return (
    <PageWrapper className="p-4 pb-24 bg-[#F1F8E9]">
      <AnimatedSection custom={0} className="mb-5">
        <h1 className="text-2xl font-poppins font-bold text-[#1B2B1D] mb-1">Government Schemes</h1>
        <p className="text-sm text-[#546E7A]">Discover agricultural subsidies and financial support.</p>
      </AnimatedSection>

      <AnimatedSection custom={1} className="mb-5">
        <div className="relative">
          <MagnifyingGlass size={20} className="absolute left-3 top-3 text-[#546E7A]" />
          <input 
            type="text" 
            placeholder="Search schemes..." 
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-white p-3 pl-10 rounded-xl outline-none text-[#1B2B1D] border border-gray-200 focus:border-[#2E7D32] focus:ring-2 focus:ring-[#2E7D32]/20 transition-all shadow-sm"
          />
        </div>
      </AnimatedSection>

      <AnimatedSection custom={2} className="mb-6">
        <div className="flex overflow-x-auto space-x-2 pb-2 -mx-4 px-4 hide-scrollbar">
          {categories.map((cat) => (
            <button
              key={cat}
              onClick={() => setActiveCategory(cat)}
              className={`whitespace-nowrap px-4 py-1.5 rounded-full text-sm font-medium transition-colors border ${
                activeCategory === cat 
                  ? 'bg-[#2E7D32] text-white border-[#2E7D32]' 
                  : 'bg-white text-[#546E7A] border-gray-200 hover:border-[#2E7D32] hover:text-[#2E7D32]'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>
      </AnimatedSection>

      <div className="space-y-4">
        {filteredSchemes.length > 0 ? (
          filteredSchemes.map((scheme, idx) => (
            <AnimatedSection custom={3 + idx * 0.1} key={scheme.id}>
              <div className="bg-white rounded-2xl shadow-sm overflow-hidden flex flex-col relative border-l-4" style={{ borderLeftColor: scheme.color }}>
                <div className="p-4 flex-1">
                  <div className="flex justify-between items-start mb-2">
                    <div className="flex items-center space-x-2">
                      <span className="text-2xl" aria-hidden="true">{scheme.emoji}</span>
                      <div>
                        <h2 className="font-poppins font-bold text-[#1B2B1D] leading-tight">{scheme.name}</h2>
                        <span className="text-xs font-medium px-2 py-0.5 rounded bg-gray-100 text-gray-600 inline-block mt-1">{scheme.category}</span>
                      </div>
                    </div>
                  </div>
                  <h3 className="text-sm text-[#546E7A] font-medium mb-3">{scheme.fullName}</h3>
                  <p className="text-sm text-[#1B2B1D] mb-4 leading-relaxed line-clamp-2">{scheme.description}</p>
                  
                  <div className="bg-[#F1F8E9] rounded-xl p-3 grid grid-cols-1 gap-2 mb-4">
                    <div className="flex justify-between items-center text-sm">
                      <span className="text-[#546E7A]">Benefit:</span>
                      <span className="font-semibold text-[#2E7D32]">{scheme.amount}</span>
                    </div>
                    <div className="flex justify-between items-start text-sm">
                      <span className="text-[#546E7A] whitespace-nowrap mr-2">Eligibility:</span>
                      <span className="font-medium text-[#1B2B1D] text-right">{scheme.eligibility}</span>
                    </div>
                  </div>

                  <div className="flex space-x-3 mt-auto">
                    <button
                      onClick={() =>
                        navigate('/chat', {
                          state: {
                            prefill: `Am I eligible for ${scheme.name} (${scheme.fullName})? My farm is in Karnataka.`,
                          },
                        })
                      }
                      className="flex-1 py-2.5 border border-[#2E7D32] text-[#2E7D32] rounded-xl text-sm font-medium hover:bg-[#E8F5E9] transition-colors flex justify-center items-center"
                    >
                      <CheckCircle weight="fill" className="mr-1.5" size={18} />
                      Check Eligibility
                    </button>
                    <button
                      onClick={() => window.open(scheme.applyLink, '_blank', 'noopener,noreferrer')}
                      className="flex-1 py-2.5 bg-[#2E7D32] text-white rounded-xl text-sm font-medium hover:bg-[#1B5E20] transition-colors flex justify-center items-center shadow-md shadow-[#2E7D32]/20"
                    >
                      Apply Now
                      <ArrowRight weight="bold" className="ml-1.5" size={16} />
                    </button>
                  </div>
                </div>
              </div>
            </AnimatedSection>
          ))
        ) : (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-center py-10">
            <div className="text-4xl mb-3">🔍</div>
            <h3 className="text-lg font-medium text-[#1B2B1D]">No schemes found</h3>
            <p className="text-[#546E7A]">Try adjusting your search or category filter</p>
          </motion.div>
        )}
      </div>
    </PageWrapper>
  );
}
