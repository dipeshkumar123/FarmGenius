export const getNumberingSystem = (lang: string): string => {
  switch (lang) {
    case 'hi':
    case 'mr':
      return 'deva'; // Devanagari numerals
    case 'ta':
      return 'tamldec'; // Tamil numerals
    case 'te':
      return 'telu'; // Telugu numerals
    case 'kn':
      return 'knda'; // Kannada numerals
    default:
      return 'latn'; // Latin/Western numerals
  }
};

export const localizeNumber = (num: number, language: string, options: Intl.NumberFormatOptions = {}): string => {
  try {
    const numberingSystem = getNumberingSystem(language);
    // Use 'en-IN' as fallback for latn to ensure correct comma formatting (e.g., 1,00,000)
    const locale = numberingSystem === 'latn' ? 'en-IN' : `${language}-IN`;
    return new Intl.NumberFormat(locale, { ...options, numberingSystem }).format(num);
  } catch (e) {
    // Fallback to standard Indian English formatting if Intl fails
    return num.toLocaleString('en-IN', options);
  }
};
