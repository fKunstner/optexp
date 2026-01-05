import asyncio

from tqdm.asyncio import tqdm as asyncio_tqdm


async def async_dummy_task(i):
    # asyncio.sleep(0.01 * i)
    i = 0
    for i in range(1_000_000):
        i += 1
    return i


async def async_main():
    for f in asyncio_tqdm.as_completed([async_dummy_task(x) for x in range(100)]):
        await f


if __name__ == "__main__":

    asyncio.run(async_main())
