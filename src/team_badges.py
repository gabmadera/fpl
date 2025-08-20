from __future__ import annotations

from typing import Dict, Optional
import logging


class TeamBadgeManager:
    """Manages team badges and visual assets for FPL teams"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # FPL Team ID to badge mapping (2025/26 season)
        self.team_badges = {
            1: {"name": "Arsenal", "short": "ARS", "badge": "ARS.png", "color": "#DC143C"},
            2: {"name": "Aston Villa", "short": "AVL", "badge": "AVL.png", "color": "#95BFE5"},
            3: {"name": "Bournemouth", "short": "BOU", "badge": "BOU.png", "color": "#DA020E"},
            4: {"name": "Brentford", "short": "BRE", "badge": "BRE.png", "color": "#FEE133"},
            5: {"name": "Brighton", "short": "BHA", "badge": "BHA.png", "color": "#0057B8"},
            6: {"name": "Chelsea", "short": "CHE", "badge": "CHE.png", "color": "#034694"},
            7: {"name": "Crystal Palace", "short": "CRY", "badge": "CRY.png", "color": "#1B458F"},
            8: {"name": "Everton", "short": "EVE", "badge": "EVE.png", "color": "#003399"},
            9: {"name": "Fulham", "short": "FUL", "badge": "FUL.png", "color": "#000000"},
            10: {"name": "Ipswich Town", "short": "IPS", "badge": "LEE.png", "color": "#4C9FE7"},  # Using LEE as closest match
            11: {"name": "Leicester City", "short": "LEI", "badge": "LEE.png", "color": "#003090"},  # Using LEE.png
            12: {"name": "Liverpool", "short": "LIV", "badge": "LIV.png", "color": "#C8102E"},
            13: {"name": "Manchester City", "short": "MCI", "badge": "MCI.png", "color": "#6CABDD"},
            14: {"name": "Manchester Utd", "short": "MUN", "badge": "MUN.png", "color": "#DA020E"},
            15: {"name": "Newcastle", "short": "NEW", "badge": "NEW.png", "color": "#241F20"},
            16: {"name": "Nottingham Forest", "short": "NFO", "badge": "NFO.png", "color": "#DD0000"},
            17: {"name": "Southampton", "short": "SOU", "badge": "SUN.png", "color": "#D71920"},  # Using SUN as closest match
            18: {"name": "Tottenham", "short": "TOT", "badge": "TOT.png", "color": "#132257"},
            19: {"name": "West Ham", "short": "WHU", "badge": "WHU.png", "color": "#7A263A"},
            20: {"name": "Wolves", "short": "WOL", "badge": "WOL.png", "color": "#FDB462"}
        }
        
        # Fallback badge for missing logos
        self.default_badge = "ARS.png"  # Use Arsenal as default fallback
        
    def get_team_badge_url(self, team_id: int, fallback_to_initials: bool = True) -> str:
        """Get badge URL for team ID"""
        try:
            if team_id in self.team_badges:
                badge_filename = self.team_badges[team_id]["badge"]
                return f"/static/badges/{badge_filename}"
            else:
                if fallback_to_initials:
                    return f"/static/badges/{self.default_badge}"
                else:
                    return None
        except Exception as e:
            self.logger.error(f"Failed to get badge for team {team_id}: {e}")
            return f"/static/badges/{self.default_badge}"
    
    def get_team_info(self, team_id: int) -> Dict[str, str]:
        """Get complete team information including badge"""
        try:
            if team_id in self.team_badges:
                info = self.team_badges[team_id].copy()
                info["badge_url"] = self.get_team_badge_url(team_id)
                return info
            else:
                return {
                    "name": f"Team {team_id}",
                    "short": f"T{team_id}",
                    "badge": self.default_badge,
                    "badge_url": f"/static/badges/{self.default_badge}",
                    "color": "#999999"
                }
        except Exception as e:
            self.logger.error(f"Failed to get team info for {team_id}: {e}")
            return {
                "name": "Unknown Team",
                "short": "UNK", 
                "badge": self.default_badge,
                "badge_url": f"/static/badges/{self.default_badge}",
                "color": "#999999"
            }
    
    def get_all_teams(self) -> Dict[int, Dict[str, str]]:
        """Get all teams with badge information"""
        all_teams = {}
        for team_id in self.team_badges:
            all_teams[team_id] = self.get_team_info(team_id)
        return all_teams
    
    def create_badge_css(self) -> str:
        """Generate CSS classes for team colors"""
        css_rules = []
        
        for team_id, info in self.team_badges.items():
            team_short = info["short"].lower()
            color = info["color"]
            
            css_rules.append(f"""
            .team-{team_short} {{
                background-color: {color};
                border-color: {color};
            }}
            
            .team-{team_short}-text {{
                color: {color};
            }}
            
            .team-{team_short}-border {{
                border: 2px solid {color};
            }}
            """)
        
        return "\n".join(css_rules)
    
    def get_formation_positions(self, formation: str) -> Dict[str, List[Dict]]:
        """Get player positions for formation visualization"""
        formation_layouts = {
            "3-4-3": {
                "GKP": [{"x": 50, "y": 85}],  # Goalkeeper
                "DEF": [  # 3 defenders
                    {"x": 25, "y": 65},
                    {"x": 50, "y": 65}, 
                    {"x": 75, "y": 65}
                ],
                "MID": [  # 4 midfielders
                    {"x": 15, "y": 40},
                    {"x": 38, "y": 40},
                    {"x": 62, "y": 40},
                    {"x": 85, "y": 40}
                ],
                "FWD": [  # 3 forwards
                    {"x": 25, "y": 15},
                    {"x": 50, "y": 15},
                    {"x": 75, "y": 15}
                ]
            },
            "3-5-2": {
                "GKP": [{"x": 50, "y": 85}],
                "DEF": [
                    {"x": 25, "y": 65},
                    {"x": 50, "y": 65},
                    {"x": 75, "y": 65}
                ],
                "MID": [
                    {"x": 15, "y": 45},
                    {"x": 35, "y": 35},
                    {"x": 50, "y": 30},
                    {"x": 65, "y": 35},
                    {"x": 85, "y": 45}
                ],
                "FWD": [
                    {"x": 35, "y": 15},
                    {"x": 65, "y": 15}
                ]
            },
            "4-3-3": {
                "GKP": [{"x": 50, "y": 85}],
                "DEF": [
                    {"x": 15, "y": 65},
                    {"x": 38, "y": 65},
                    {"x": 62, "y": 65},
                    {"x": 85, "y": 65}
                ],
                "MID": [
                    {"x": 30, "y": 40},
                    {"x": 50, "y": 40},
                    {"x": 70, "y": 40}
                ],
                "FWD": [
                    {"x": 25, "y": 15},
                    {"x": 50, "y": 15},
                    {"x": 75, "y": 15}
                ]
            },
            "4-4-2": {
                "GKP": [{"x": 50, "y": 85}],
                "DEF": [
                    {"x": 15, "y": 65},
                    {"x": 38, "y": 65},
                    {"x": 62, "y": 65},
                    {"x": 85, "y": 65}
                ],
                "MID": [
                    {"x": 20, "y": 40},
                    {"x": 40, "y": 40},
                    {"x": 60, "y": 40},
                    {"x": 80, "y": 40}
                ],
                "FWD": [
                    {"x": 35, "y": 15},
                    {"x": 65, "y": 15}
                ]
            },
            "4-5-1": {
                "GKP": [{"x": 50, "y": 85}],
                "DEF": [
                    {"x": 15, "y": 65},
                    {"x": 38, "y": 65},
                    {"x": 62, "y": 65},
                    {"x": 85, "y": 65}
                ],
                "MID": [
                    {"x": 15, "y": 45},
                    {"x": 32, "y": 35},
                    {"x": 50, "y": 30},
                    {"x": 68, "y": 35},
                    {"x": 85, "y": 45}
                ],
                "FWD": [
                    {"x": 50, "y": 15}
                ]
            },
            "5-3-2": {
                "GKP": [{"x": 50, "y": 85}],
                "DEF": [
                    {"x": 10, "y": 65},
                    {"x": 30, "y": 65},
                    {"x": 50, "y": 65},
                    {"x": 70, "y": 65},
                    {"x": 90, "y": 65}
                ],
                "MID": [
                    {"x": 30, "y": 40},
                    {"x": 50, "y": 40},
                    {"x": 70, "y": 40}
                ],
                "FWD": [
                    {"x": 35, "y": 15},
                    {"x": 65, "y": 15}
                ]
            },
            "5-4-1": {
                "GKP": [{"x": 50, "y": 85}],
                "DEF": [
                    {"x": 10, "y": 65},
                    {"x": 30, "y": 65},
                    {"x": 50, "y": 65},
                    {"x": 70, "y": 65},
                    {"x": 90, "y": 65}
                ],
                "MID": [
                    {"x": 22, "y": 40},
                    {"x": 42, "y": 40},
                    {"x": 58, "y": 40},
                    {"x": 78, "y": 40}
                ],
                "FWD": [
                    {"x": 50, "y": 15}
                ]
            }
        }
        
        return formation_layouts.get(formation, formation_layouts["3-4-3"])
    
    def generate_default_badges(self) -> Dict[str, str]:
        """Generate SVG badges for teams if PNG files don't exist"""
        svg_badges = {}
        
        for team_id, info in self.team_badges.items():
            short_name = info["short"]
            color = info["color"]
            
            svg_content = f'''
            <svg width="40" height="40" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
                <circle cx="20" cy="20" r="18" fill="{color}" stroke="#fff" stroke-width="2"/>
                <text x="20" y="26" text-anchor="middle" fill="white" font-family="Arial, sans-serif" font-size="8" font-weight="bold">
                    {short_name}
                </text>
            </svg>
            '''
            
            svg_badges[f"{short_name.lower()}.svg"] = svg_content
        
        return svg_badges