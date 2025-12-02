import asyncio
from understat import Understat
import aiohttp

async def test():
    async with aiohttp.ClientSession() as session:
        understat = Understat(session)
        
        # ลองดึง player data ของ Salah
        player_data = await understat.get_player_stats(1250)
        print("✓ Understat API works!")
        print(f"✓ Sample data: {player_data[:2]}")
        print(f"✓ Total seasons: {len(player_data)}")

asyncio.run(test())
