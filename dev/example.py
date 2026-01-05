import asyncio

from tqdm.asyncio import tqdm


async def main():
    for i in tqdm.as_completed([asyncio.sleep(0.01 * i) for i in range(100)]):
        await i


if __name__ == "__main__":
    asyncio.run(main())
