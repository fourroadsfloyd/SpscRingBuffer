#ifndef SPSC_RING_BUFFER_H
#define SPSC_RING_BUFFER_H

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <spdlog/spdlog.h>

/**
 * @brief 单生产者单消费者（SPSC）无锁环形缓冲区。
 *
 * 该容器仅支持一个生产者线程写入、一个消费者线程读取。
 * SIZE 必须是 2 的幂，便于通过位与快速取模。
 *
 * @tparam T 元素类型，需满足平凡可拷贝（trivially copyable）。
 * @tparam SIZE 缓冲区容量（元素个数，且必须是 2 的幂）。
 */
template <typename T, size_t SIZE>
class SpscRingBuffer
{
public:
    static_assert((SIZE & (SIZE - 1)) == 0, "SpscRingBuffer size must be a power of 2");
    static_assert(std::is_trivially_copyable<T>::value,
                  "SpscRingBuffer requires trivially copyable T");

    static constexpr size_t kMask = SIZE - 1;

    SpscRingBuffer() = default;
    ~SpscRingBuffer() = default;

    SpscRingBuffer(const SpscRingBuffer&) = delete;
    SpscRingBuffer& operator=(const SpscRingBuffer&) = delete;
    SpscRingBuffer(SpscRingBuffer&&) = delete;
    SpscRingBuffer& operator=(SpscRingBuffer&&) = delete;

    /**
     * @brief 写入一个元素。
     * @param item 待写入元素。
     * @return true 写入成功；false 表示缓冲区已满。
     */
    bool push(const T& item)
    {
        const size_t tail = m_tail.load(std::memory_order_relaxed);
        const size_t head = m_head.load(std::memory_order_acquire);
        if ((tail - head) == SIZE)
        {
            SPDLOG_WARN("SpscRingBuffer is full, cannot push item");
            return false;
        }

        m_buffer[tail & kMask] = item;
        m_tail.store(tail + 1, std::memory_order_release);
        return true;
    }

    /**
     * @brief 读取一个元素。
     * @param item 读取结果输出参数。
     * @return true 读取成功；false 表示缓冲区为空。
     */
    bool pop(T& item)
    {
        const size_t head = m_head.load(std::memory_order_relaxed);
        const size_t tail = m_tail.load(std::memory_order_acquire);
        if (head == tail)
        {
            SPDLOG_WARN("SpscRingBuffer is empty, cannot pop item");
            return false;
        }

        item = m_buffer[head & kMask];
        m_head.store(head + 1, std::memory_order_release);
        return true;
    }


    /**
     * @brief 批量写入元素。
     * @param data 输入数据起始地址。
     * @param count 期望写入元素个数。
     * @return 实际写入元素个数。
     */
    size_t push(const T* data, size_t count)
    {
        if (data == nullptr || count == 0)
        {
            SPDLOG_WARN("SpscRingBuffer push called with null data or zero count");
            return 0;
        }

        const size_t tail = m_tail.load(std::memory_order_relaxed);
        const size_t head = m_head.load(std::memory_order_acquire);
        const size_t space = SIZE - (tail - head);
        const size_t writeCount = std::min(count, space);
        if (writeCount == 0)
        {
            SPDLOG_WARN("SpscRingBuffer is full, cannot push items");
            return 0;
        }

        const size_t tailPos = tail & kMask;
        const size_t firstPart = std::min(writeCount, SIZE - tailPos);
        std::memcpy(&m_buffer[tailPos], data, firstPart * sizeof(T));

        const size_t secondPart = writeCount - firstPart;
        if (secondPart > 0)
        {
            std::memcpy(&m_buffer[0], data + firstPart, secondPart * sizeof(T));
        }

        m_tail.store(tail + writeCount, std::memory_order_release);
        return writeCount;
    }


    /**
     * @brief 批量读取元素。
     * @param data 输出数据起始地址。
     * @param count 期望读取元素个数。
     * @return 实际读取元素个数。
     */
    size_t pop(T* data, size_t count)
    {
        if (data == nullptr || count == 0)
        {
            SPDLOG_WARN("SpscRingBuffer pop called with null data or zero count");
            return 0;
        }

        const size_t head = m_head.load(std::memory_order_relaxed);
        const size_t tail = m_tail.load(std::memory_order_acquire);
        const size_t dataCount = tail - head;
        const size_t readCount = std::min(count, dataCount);
        if (readCount == 0)
        {
            SPDLOG_WARN("SpscRingBuffer is empty, cannot pop items");
            return 0;
        }

        const size_t headPos = head & kMask;
        const size_t firstPart = std::min(readCount, SIZE - headPos);
        std::memcpy(data, &m_buffer[headPos], firstPart * sizeof(T));

        const size_t secondPart = readCount - firstPart;
        if (secondPart > 0)
        {
            std::memcpy(data + firstPart, &m_buffer[0], secondPart * sizeof(T));
        }

        m_head.store(head + readCount, std::memory_order_release);
        return readCount;
    }

    /**
     * @brief 获取当前可读元素个数。
     * @return 可读元素个数。
     */
    size_t availableData() const
    {
        const size_t head = m_head.load(std::memory_order_acquire);
        const size_t tail = m_tail.load(std::memory_order_acquire);
        return tail - head;
    }

    /**
     * @brief 获取当前可写元素个数。
     * @return 可写元素个数。
     */
    size_t availableSpace() const
    {
        return SIZE - availableData();
    }

    /**
     * @brief 检查缓冲区是否为空。
     * @return true 为空；false 非空。
     */
    bool isEmpty() const
    {
        return availableData() == 0;
    }

    /**
     * @brief 检查缓冲区是否已满。
     * @return true 已满；false 未满。
     */
    bool isFull() const
    {
        return availableData() == SIZE;
    }

    /**
     * @brief 获取总容量（元素个数）。
     * @return 固定容量 SIZE。
     */
    constexpr size_t capacity() const
    {
        return SIZE;
    }

    /**
     * @brief 获取当前读指针位置（按容量取模后的下标）。
     * @return 读指针下标。
     */
    size_t head() const
    {
        return m_head.load(std::memory_order_acquire) & kMask;
    }

    /**
     * @brief 获取当前写指针位置（按容量取模后的下标）。
     * @return 写指针下标。
     */
    size_t tail() const
    {
        return m_tail.load(std::memory_order_acquire) & kMask;
    }

    /**
     * @brief 获取底层连续存储地址。
     * @return 可写缓冲区首地址。
     */
    T* data()
    {
        return m_buffer.data();
    }

    /**
     * @brief 获取底层连续存储地址（只读）。
     * @return 只读缓冲区首地址。
     */
    const T* data() const
    {
        return m_buffer.data();
    }

    /**
     * @brief 消费指定数量的数据（仅消费者线程调用）。
     * @param dataSize 需要消费的数据量。
     */
    void delDataSize(size_t dataSize)
    {
        if (dataSize == 0)
        {
            return;
        }

        const size_t head = m_head.load(std::memory_order_relaxed);
        const size_t tail = m_tail.load(std::memory_order_acquire);
        size_t actualDataSize = dataSize;
        const size_t currentData = tail - head;
        if (actualDataSize > currentData)
        {
            actualDataSize = currentData;
        }
        m_head.store(head + actualDataSize, std::memory_order_release);
    }

    /**
     * @brief 清空缓冲区（需确保没有并发读写）。
     */
    void clear()
    {
        m_head.store(0, std::memory_order_relaxed);
        m_tail.store(0, std::memory_order_relaxed);
    }

private:
    std::array<T, SIZE> m_buffer{};
    std::atomic<size_t> m_head{0};
    std::atomic<size_t> m_tail{0};
};

#endif // SPSC_RING_BUFFER_H
