import functools
import inspect
from collections.abc import Callable
from typing import Any


def i_after(run_before_str: str) -> Callable:
    """
    实例的同步装饰器：调用目标实例方法前，需要执行另一个实例方法
    Args:
		run_before_str : 执行前需要执行的函数的名称
    """

    def decorator(func: Callable) -> Callable:
        # 检测被装饰函数是否为同步生成器函数（含 yield）
        _is_sync_gen = inspect.isgeneratorfunction(func)

        if _is_sync_gen:
            @functools.wraps(func)
            async def gen_wrapper(self, *args, **kwargs):
                isinstance(self, object)

                run_before = getattr(self, run_before_str, None)
                if run_before is None or not callable(run_before):
                    raise AttributeError(
                        f"{self.__class__.__name__} 没有可调用的 {run_before_str}"
                    )
                if inspect.iscoroutinefunction(run_before):
                    raise TypeError(f"{run_before} 必须是同步函数")

                run_before()

                for item in func(self, *args, **kwargs):
                    yield item

            return gen_wrapper

        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs) -> Any:
            # 严格保证 self 是实例对象
            isinstance(self, object)

            # 被装饰方法必须是同步函数（非协程）
            if inspect.iscoroutinefunction(func):
                raise TypeError(f"被装饰的方法 {func.__name__} 必须是同步函数")

            # 获取前置方法
            run_before = getattr(self, run_before_str, None)

            # 前置方法非空与函数类型检查
            if run_before is None or not callable(run_before):
                raise AttributeError(f"{self.__class__.__name__} 没有可调用的 {run_before_str}")

            # 前置方法必须是同步函数
            if inspect.iscoroutinefunction(run_before):
                raise TypeError(f"{run_before} 必须是同步函数")

            run_before()

            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def i_aafter(arun_before_str: str):
    """
    实例的同步装饰器：调用目标实例方法前，需要执行另一个实例方法
    Args:
        arun_before_str : 执行前需要执行的函数的名称
    """

    def decorator(func: Callable) -> Callable:
        # 检测被装饰函数是否为异步生成器函数（含 yield）
        _is_async_gen = inspect.isasyncgenfunction(func)

        if _is_async_gen:
            @functools.wraps(func)
            async def gen_wrapper(self, *args, **kwargs):
                isinstance(self, object)

                arun_before = getattr(self, arun_before_str, None)
                if arun_before is None or not callable(arun_before):
                    raise AttributeError(
                        f"{self.__class__.__name__} 没有可调用的 {arun_before_str}"
                    )
                if not inspect.iscoroutinefunction(arun_before):
                    raise TypeError(f"{arun_before_str} 必须是 async 函数")

                await arun_before()

                async for item in func(self, *args, **kwargs):
                    yield item

            return gen_wrapper

        @functools.wraps(func)
        async def wrapper(self, *args, **kwargs) -> Any:
            isinstance(self, object)

            if not inspect.iscoroutinefunction(func):
                raise TypeError(f"被装饰的方法 {func.__name__} 必须是 async 函数")

            arun_before = getattr(self, arun_before_str, None)
            if arun_before is None or not callable(arun_before):
                raise AttributeError(
                    f"{self.__class__.__name__} 没有可调用的 {arun_before_str}"
                )
            if not inspect.iscoroutinefunction(arun_before):
                raise TypeError(f"{arun_before_str} 必须是 async 函数")

            await arun_before()

            return await func(self, *args, **kwargs)

        return wrapper

    return decorator
