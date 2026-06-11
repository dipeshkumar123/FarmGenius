import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:flutter_animate/flutter_animate.dart';
import 'package:google_fonts/google_fonts.dart';

// ─────────────────────────────────────────────
// DATA MODEL
// ─────────────────────────────────────────────
class _DayForecast {
  final String day;
  final String icon;
  final int temp;

  const _DayForecast({required this.day, required this.icon, required this.temp});
}

const List<_DayForecast> _forecast = [
  _DayForecast(day: 'Mon', icon: '⛅', temp: 31),
  _DayForecast(day: 'Tue', icon: '☁', temp: 29),
  _DayForecast(day: 'Wed', icon: '🌧', temp: 27),
  _DayForecast(day: 'Thu', icon: '🌧', temp: 27),
  _DayForecast(day: 'Fri', icon: '☀️', temp: 34),
  _DayForecast(day: 'Sat', icon: '☀️', temp: 35),
  _DayForecast(day: 'Sun', icon: '⛅', temp: 33),
];

const List<double> _rainfall = [0, 2, 18, 25, 3, 0, 1];
const List<String> _rainfallDays = ['M', 'T', 'W', 'T', 'F', 'S', 'S'];

class _Advisory {
  final String icon;
  final String title;
  final String body;
  final Color accentColor;

  const _Advisory({
    required this.icon,
    required this.title,
    required this.body,
    required this.accentColor,
  });
}

const List<_Advisory> _advisories = [
  _Advisory(
    icon: '🌧️',
    title: 'Rain expected Thu–Fri',
    body: 'Avoid pesticide spray on Wednesday evening.',
    accentColor: Color(0xFF1565C0),
  ),
  _Advisory(
    icon: '🌡️',
    title: 'High temperature next weekend',
    body: 'Ensure adequate irrigation for standing crops.',
    accentColor: Color(0xFFE65100),
  ),
  _Advisory(
    icon: '✅',
    title: 'Good sowing window Mon–Wed',
    body: 'Favorable conditions for kharif sowing.',
    accentColor: Color(0xFF2E7D32),
  ),
];

// ─────────────────────────────────────────────
// SCREEN
// ─────────────────────────────────────────────
class WeatherScreen extends StatelessWidget {
  const WeatherScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFFF1F8E9),
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        centerTitle: false,
        title: Text(
          '🌤️ Weather',
          style: GoogleFonts.poppins(
            color: const Color(0xFF1B2B1D),
            fontWeight: FontWeight.w700,
            fontSize: 20,
          ),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.my_location_rounded, color: Color(0xFF1565C0)),
            onPressed: () {},
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.only(bottom: 32),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // ── A: HERO CARD ────────────────────────────────
            const _HeroWeatherCard()
                .animate()
                .fadeIn(duration: 400.ms)
                .slideY(begin: -0.2, curve: Curves.easeOut),

            // ── B: 7-DAY FORECAST ────────────────────────────
            Padding(
              padding: const EdgeInsets.only(left: 16, top: 20, bottom: 10),
              child: Text(
                '7-Day Forecast',
                style: GoogleFonts.poppins(
                  fontSize: 17,
                  fontWeight: FontWeight.w700,
                  color: const Color(0xFF1B2B1D),
                ),
              ),
            ),
            SizedBox(
              height: 110,
              child: ListView.builder(
                scrollDirection: Axis.horizontal,
                padding: const EdgeInsets.symmetric(horizontal: 16),
                itemCount: _forecast.length,
                itemBuilder: (context, index) {
                  final isToday = index == 0;
                  return _ForecastDayCard(
                    forecast: _forecast[index],
                    isSelected: isToday,
                    index: index,
                  );
                },
              ),
            ),

            // ── C: FARMING ADVISORIES ────────────────────────
            Padding(
              padding: const EdgeInsets.only(left: 16, top: 20, bottom: 10),
              child: Text(
                '🌾 Farming Advisories',
                style: GoogleFonts.poppins(
                  fontSize: 17,
                  fontWeight: FontWeight.w700,
                  color: const Color(0xFF1B2B1D),
                ),
              ),
            ),
            ..._advisories.asMap().entries.map(
              (e) => _AdvisoryCard(advisory: e.value, index: e.key),
            ),

            // ── D: PRECIPITATION CHART ───────────────────────
            Padding(
              padding: const EdgeInsets.only(left: 16, top: 20, bottom: 10),
              child: Text(
                '💧 Weekly Rainfall (mm)',
                style: GoogleFonts.poppins(
                  fontSize: 17,
                  fontWeight: FontWeight.w700,
                  color: const Color(0xFF1B2B1D),
                ),
              ),
            ),
            const _RainfallChart(),
          ],
        ),
      ),
    );
  }
}

// ─────────────────────────────────────────────
// A — HERO WEATHER CARD
// ─────────────────────────────────────────────
class _HeroWeatherCard extends StatelessWidget {
  const _HeroWeatherCard();

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.fromLTRB(16, 16, 16, 0),
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFF1565C0), Color(0xFF42A5F5)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(24),
        boxShadow: [
          BoxShadow(
            color: const Color(0xFF1565C0).withValues(alpha: 0.40),
            blurRadius: 24,
            offset: const Offset(0, 10),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Location row
          Row(
            children: [
              const Icon(Icons.location_on, color: Colors.white70, size: 16),
              const SizedBox(width: 4),
              Text(
                'Bengaluru, Karnataka',
                style: GoogleFonts.inter(
                  color: Colors.white70,
                  fontSize: 13,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          // Temp + icon row
          Row(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      '32°C',
                      style: GoogleFonts.poppins(
                        color: Colors.white,
                        fontSize: 52,
                        fontWeight: FontWeight.w700,
                        height: 1.0,
                      ),
                    ),
                    Text(
                      'Partly Cloudy',
                      style: GoogleFonts.inter(
                        color: Colors.white70,
                        fontSize: 16,
                      ),
                    ),
                    const SizedBox(height: 14),
                    // Metrics row
                    Wrap(
                      spacing: 16,
                      runSpacing: 8,
                      children: [
                        _MetricChip(label: '💧 65%', sublabel: 'Humidity'),
                        _MetricChip(label: '💨 14 km/h', sublabel: 'Wind'),
                        _MetricChip(label: 'UV 7', sublabel: 'High'),
                      ],
                    ),
                  ],
                ),
              ),
              const Icon(
                Icons.wb_sunny_rounded,
                color: Colors.amber,
                size: 72,
              ),
            ],
          ),
          const SizedBox(height: 14),
          const Divider(color: Colors.white24, thickness: 1),
          const SizedBox(height: 10),
          // Sunrise / Sunset
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,
            children: [
              _SunriseSunset(icon: '🌅', label: 'Sunrise', time: '06:12'),
              Container(
                width: 1,
                height: 30,
                color: Colors.white24,
              ),
              _SunriseSunset(icon: '🌇', label: 'Sunset', time: '18:45'),
            ],
          ),
        ],
      ),
    );
  }
}

class _MetricChip extends StatelessWidget {
  final String label;
  final String sublabel;

  const _MetricChip({required this.label, required this.sublabel});

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          label,
          style: GoogleFonts.inter(
            color: Colors.white,
            fontWeight: FontWeight.w600,
            fontSize: 13,
          ),
        ),
        Text(
          sublabel,
          style: GoogleFonts.inter(
            color: Colors.white60,
            fontSize: 11,
          ),
        ),
      ],
    );
  }
}

class _SunriseSunset extends StatelessWidget {
  final String icon;
  final String label;
  final String time;

  const _SunriseSunset({
    required this.icon,
    required this.label,
    required this.time,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Text(icon, style: const TextStyle(fontSize: 20)),
        const SizedBox(height: 4),
        Text(
          label,
          style: GoogleFonts.inter(color: Colors.white60, fontSize: 11),
        ),
        Text(
          time,
          style: GoogleFonts.poppins(
            color: Colors.white,
            fontWeight: FontWeight.w600,
            fontSize: 14,
          ),
        ),
      ],
    );
  }
}

// ─────────────────────────────────────────────
// B — FORECAST DAY CARD
// ─────────────────────────────────────────────
class _ForecastDayCard extends StatelessWidget {
  final _DayForecast forecast;
  final bool isSelected;
  final int index;

  const _ForecastDayCard({
    required this.forecast,
    required this.isSelected,
    required this.index,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: 80,
      margin: const EdgeInsets.only(right: 10),
      decoration: BoxDecoration(
        color: isSelected ? const Color(0xFF2E7D32) : Colors.white,
        borderRadius: BorderRadius.circular(12),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.06),
            blurRadius: 8,
            offset: const Offset(0, 3),
          ),
        ],
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(
            forecast.day,
            style: GoogleFonts.inter(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: isSelected ? Colors.white : Colors.grey[600],
            ),
          ),
          const SizedBox(height: 8),
          Text(forecast.icon, style: const TextStyle(fontSize: 24)),
          const SizedBox(height: 8),
          Text(
            '${forecast.temp}°',
            style: GoogleFonts.poppins(
              fontSize: 15,
              fontWeight: FontWeight.w700,
              color: isSelected ? Colors.white : const Color(0xFF1B2B1D),
            ),
          ),
        ],
      ),
    )
        .animate()
        .fadeIn(delay: (index * 100).ms, duration: 350.ms)
        .slideX(begin: 0.2, curve: Curves.easeOut);
  }
}

// ─────────────────────────────────────────────
// C — ADVISORY CARD
// ─────────────────────────────────────────────
class _AdvisoryCard extends StatelessWidget {
  final _Advisory advisory;
  final int index;

  const _AdvisoryCard({required this.advisory, required this.index});

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 16, vertical: 5),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border(
          left: BorderSide(color: advisory.accentColor, width: 3),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 10,
            offset: const Offset(0, 3),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(advisory.icon, style: const TextStyle(fontSize: 18)),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  advisory.title,
                  style: GoogleFonts.poppins(
                    fontWeight: FontWeight.w600,
                    fontSize: 14,
                    color: const Color(0xFF1B2B1D),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 6),
          Text(
            advisory.body,
            style: GoogleFonts.inter(
              fontSize: 14,
              color: Colors.grey[600],
              height: 1.4,
            ),
          ),
        ],
      ),
    )
        .animate()
        .fadeIn(delay: (index * 120).ms, duration: 350.ms)
        .slideX(begin: -0.2, curve: Curves.easeOut);
  }
}

// ─────────────────────────────────────────────
// D — RAINFALL BAR CHART
// ─────────────────────────────────────────────
class _RainfallChart extends StatelessWidget {
  const _RainfallChart();

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 180,
      margin: const EdgeInsets.symmetric(horizontal: 16),
      padding: const EdgeInsets.fromLTRB(12, 16, 12, 8),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 10,
            offset: const Offset(0, 3),
          ),
        ],
      ),
      child: BarChart(
        BarChartData(
          maxY: 30,
          minY: 0,
          barTouchData: BarTouchData(enabled: true),
          gridData: FlGridData(
            show: true,
            drawVerticalLine: false,
            horizontalInterval: 10,
            getDrawingHorizontalLine: (_) => FlLine(
              color: const Color(0xFFE8F5E9),
              strokeWidth: 1,
            ),
          ),
          borderData: FlBorderData(show: false),
          titlesData: FlTitlesData(
            leftTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                reservedSize: 28,
                interval: 10,
                getTitlesWidget: (value, meta) {
                  if (value == 0 || value == 10 || value == 20 || value == 30) {
                    return Text(
                      value.toInt().toString(),
                      style: GoogleFonts.inter(
                        fontSize: 10,
                        color: Colors.grey[500],
                      ),
                    );
                  }
                  return const SizedBox.shrink();
                },
              ),
            ),
            bottomTitles: AxisTitles(
              sideTitles: SideTitles(
                showTitles: true,
                reservedSize: 22,
                getTitlesWidget: (value, meta) {
                  final i = value.toInt();
                  if (i < 0 || i >= _rainfallDays.length) {
                    return const SizedBox.shrink();
                  }
                  return Text(
                    _rainfallDays[i],
                    style: GoogleFonts.inter(
                      fontSize: 11,
                      fontWeight: FontWeight.w600,
                      color: Colors.grey[600],
                    ),
                  );
                },
              ),
            ),
            topTitles: const AxisTitles(
              sideTitles: SideTitles(showTitles: false),
            ),
            rightTitles: const AxisTitles(
              sideTitles: SideTitles(showTitles: false),
            ),
          ),
          barGroups: _rainfall.asMap().entries.map((e) {
            final isHighRain = e.value >= 10;
            return BarChartGroupData(
              x: e.key,
              barRods: [
                BarChartRodData(
                  toY: e.value,
                  color: isHighRain
                      ? const Color(0xFF1565C0)
                      : const Color(0xFF42A5F5),
                  width: 18,
                  borderRadius: const BorderRadius.vertical(
                    top: Radius.circular(6),
                  ),
                  backDrawRodData: BackgroundBarChartRodData(
                    show: true,
                    toY: 30,
                    color: const Color(0xFFF1F8E9),
                  ),
                ),
              ],
            );
          }).toList(),
        ),
      ),
    ).animate().fadeIn(delay: 200.ms, duration: 500.ms).slideY(begin: 0.15);
  }
}
