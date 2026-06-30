import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { PageWrapper, AnimatedSection } from '../components/ui/PageWrapper';
import { MagnifyingGlass, CheckCircle, ArrowRight, Spinner } from 'phosphor-react';
import { motion } from 'framer-motion';
import apiClient from '../api/client';
import { useAppStore } from '../store/appStore';
import { useTranslation } from 'react-i18next';
import { localizeNumber } from '../utils/localization';

const categories = ['All', 'Direct Benefit', 'Crop Insurance', 'Credit/Loans', 'Irrigation', 'Market Access', 'Government'];

export default function SchemesPage() {
  const navigate = useNavigate();
  const { t, i18n } = useTranslation();
  const farmer = useAppStore((s) => s.farmer);
  const isOffline = useAppStore((s) => s.isOffline);

  const [search, setSearch] = useState('');
  const [activeCategory, setActiveCategory] = useState('All');
  const [schemes, setSchemes] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchSchemes = async () => {
      setLoading(true);
      try {
        const crop = 'wheat'; // general search
        const state = farmer?.state ?? 'Karnataka';
        const res = await apiClient.get('/schemes', { params: { crop, state } });
        
        const liveSchemes = res.data.map((item: any, idx: number) => ({
          id: `live-${idx}`,
          name: item.scheme_name,
          fullName: item.scheme_name,
          category: item.category || 'Government',
          amount: item.amount || t('schemes.portal'),
          eligibility: item.eligibility,
          deadline: t('schemes.website'),
          color: '#2E7D32',
          emoji: '🏛️',
          description: item.description,
          applyLink: item.link
        }));

        setSchemes(liveSchemes);
      } catch (error) {
        console.error("Error fetching schemes", error);
        setSchemes([]);
      } finally {
        setLoading(false);
      }
    };
    
    if (!isOffline) {
      fetchSchemes();
    } else {
      setSchemes([]);
      setLoading(false);
    }
  }, [farmer?.state, isOffline]);

  const filteredSchemes = schemes.filter(s => {
    const matchesSearch = s.name.toLowerCase().includes(search.toLowerCase()) || s.fullName.toLowerCase().includes(search.toLowerCase());
    const matchesCategory = activeCategory === 'All' || s.category === activeCategory;
    return matchesSearch && matchesCategory;
  });

  const handleApply = (link: string) => {
    if (link && (link.startsWith('http://') || link.startsWith('https://'))) {
      window.open(link, '_blank', 'noopener,noreferrer');
    } else {
      alert(t('schemes.portal')); // Fallback message if link is missing or invalid
    }
  };

  return (
    <PageWrapper className="p-4 pb-24 bg-[#F1F8E9]">
      <AnimatedSection custom={0} className="mb-5">
        <h1 className="text-2xl font-poppins font-bold text-[#1B2B1D] mb-1">{t('schemes.title')}</h1>
        <p className="text-sm text-[#546E7A]">{t('schemes.subtitle')}</p>
      </AnimatedSection>

      <AnimatedSection custom={1} className="mb-5">
        <div className="relative">
          <MagnifyingGlass size={20} className="absolute left-3 top-3 text-[#546E7A]" />
          <input 
            type="text" 
            placeholder={t('schemes.search')} 
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
              {t(`schemes.categories.${cat}` as any)}
            </button>
          ))}
        </div>
      </AnimatedSection>

      {loading ? (
        <div className="flex justify-center items-center py-10 text-[#2E7D32]">
          <Spinner className="animate-spin" size={32} />
          <span className="ml-3 font-medium">{t('schemes.loading')}</span>
        </div>
      ) : (
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
                          <span className="text-xs font-medium px-2 py-0.5 rounded bg-gray-100 text-gray-600 inline-block mt-1">
                            {t(`schemes.categories.${scheme.category}` as any, { defaultValue: scheme.category })}
                          </span>
                        </div>
                      </div>
                    </div>
                    <h3 className="text-sm text-[#546E7A] font-medium mb-3">{scheme.fullName}</h3>
                    <p className="text-sm text-[#1B2B1D] mb-4 leading-relaxed line-clamp-2">{scheme.description}</p>
                    
                    <div className="bg-[#F1F8E9] rounded-xl p-3 grid grid-cols-1 gap-2 mb-4">
                      <div className="flex justify-between items-center text-sm">
                        <span className="text-[#546E7A]">{t('schemes.benefit')}</span>
                        <span className="font-semibold text-[#2E7D32]">
                          {typeof scheme.amount === 'string' ? scheme.amount.replace(/[\d,]+/g, (match: string) => localizeNumber(parseInt(match.replace(/,/g, ''), 10), i18n.language)) : scheme.amount}
                        </span>
                      </div>
                      <div className="flex justify-between items-start text-sm">
                        <span className="text-[#546E7A] whitespace-nowrap mr-2">{t('schemes.eligibility')}</span>
                        <span className="font-medium text-[#1B2B1D] text-right">{scheme.eligibility}</span>
                      </div>
                    </div>

                    <div className="flex space-x-3 mt-auto">
                      <button
                        onClick={() =>
                          navigate('/chat', {
                            state: {
                              prefill: t('schemes.chat_prefill', {
                                name: scheme.name,
                                fullName: scheme.fullName,
                                state: farmer?.state || 'Karnataka'
                              }),
                            },
                          })
                        }
                        className="flex-1 py-2.5 border border-[#2E7D32] text-[#2E7D32] rounded-xl text-sm font-medium hover:bg-[#E8F5E9] transition-colors flex justify-center items-center"
                      >
                        <CheckCircle weight="fill" className="mr-1.5" size={18} />
                        {t('schemes.check_eligibility')}
                      </button>
                      <button
                        onClick={() => handleApply(scheme.applyLink)}
                        className="flex-1 py-2.5 bg-[#2E7D32] text-white rounded-xl text-sm font-medium hover:bg-[#1B5E20] transition-colors flex justify-center items-center shadow-md shadow-[#2E7D32]/20"
                      >
                        {t('schemes.apply_now')}
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
              <h3 className="text-lg font-medium text-[#1B2B1D]">{t('schemes.no_schemes')}</h3>
              <p className="text-[#546E7A]">{t('schemes.try_adjusting')}</p>
            </motion.div>
          )}
        </div>
      )}
    </PageWrapper>
  );
}
